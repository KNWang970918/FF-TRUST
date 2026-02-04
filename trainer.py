import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer

from models.model import Model
from datasets.dataset import LoadDataset
from evaluator import Evaluator

from torch.nn import CrossEntropyLoss
from losses.double_alignment import CORAL
from losses.ae_loss import AELoss
from torch.utils.tensorboard import SummaryWriter



class ForwardFreeRobustCELoss(nn.Module):
    def __init__(self, num_classes: int, lambda_infomax: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_infomax = lambda_infomax
        self.eps = eps

    def forward(self, logits: torch.Tensor, noisy_targets: torch.Tensor, sample_weights: torch.Tensor = None):
        if logits.dim() == 3:
            B, L, C = logits.shape
            logits = logits.reshape(B * L, C)
            noisy_targets = noisy_targets.reshape(B * L)
            if sample_weights is not None:
                sample_weights = sample_weights.reshape(B * L)

        ce = F.cross_entropy(logits, noisy_targets, reduction="none")
        if sample_weights is not None:
            ce = ce * sample_weights
        ce_loss = ce.mean()

        p = torch.softmax(logits, dim=-1).clamp(self.eps, 1.0)  

        H_each = -(p * torch.log(p)).sum(dim=-1).mean()

        p_bar = p.mean(dim=0)
        p_bar = (p_bar / p_bar.sum()).clamp(self.eps, 1.0)
        H_bar = -(p_bar * torch.log(p_bar)).sum()

        mi_reg = H_each - H_bar
        loss = ce_loss + self.lambda_infomax * mi_reg

        return loss, ce_loss.detach(), mi_reg.detach()


class ELRBuffer:
    def __init__(self, num_samples: int, num_classes: int, momentum: float = 0.9, device: str = "cuda"):
        self.P = torch.zeros(num_samples, num_classes, device=device)
        self.m = momentum

    @torch.no_grad()
    def update(self, idx: torch.Tensor, probs: torch.Tensor):
        idx = idx.long()
        self.P[idx] = self.m * self.P[idx] + (1 - self.m) * probs

    def get(self, idx: torch.Tensor) -> torch.Tensor:
        return self.P[idx.long()]


def elr_regularizer(probs: torch.Tensor, P_hist: torch.Tensor, lambda_elr: float) -> torch.Tensor:
    align = (probs * P_hist).sum(dim=-1).clamp(0, 1 - 1e-6)  
    return -lambda_elr * torch.log(1 - align).mean()

class FourierBuffer:
    def __init__(self, num_samples: int, momentum: float = 0.9, device: str = "cuda"):
        self.num_samples = num_samples
        self.m = momentum
        self.device = device
        self.P = None  

    def _lazy_init(self, mu_f_mag: torch.Tensor):
        if self.P is None:
            _, Freq, D = mu_f_mag.shape
            self.P = torch.zeros(self.num_samples, Freq, D, device=self.device)

    @torch.no_grad()
    def update(self, idx: torch.Tensor, mu_f_mag: torch.Tensor):
        idx = idx.long()
        self._lazy_init(mu_f_mag)
        mu_f_mag = mu_f_mag.detach().contiguous()
        self.P[idx] = self.m * self.P[idx] + (1 - self.m) * mu_f_mag

    def get(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.long()
        return self.P[idx]


def fourier_elr_regularizer(mu_f_mag: torch.Tensor, P_hist_f: torch.Tensor, lambda_fourier: float) -> torch.Tensor:
    """
    NOTE: rfft output often non-contiguous, avoid view; use reshape.
    """
    B = mu_f_mag.size(0)
    x = F.normalize(mu_f_mag.reshape(B, -1), dim=-1)
    y = F.normalize(P_hist_f.reshape(B, -1), dim=-1)
    cos = (x * y).sum(-1)
    cos = cos.clamp(min=0.0)           
    return lambda_fourier * (1.0 - cos).mean()
    #align = (x * y).sum(dim=-1).clamp(0, 1 - 1e-6)  
    # #return -lambda_fourier * torch.log(1 - align).mean()

class Trainer(object):
    def __init__(self, params):
        self.params = params

        ds_obj = LoadDataset(params)
        self.data_loader, subject_id = ds_obj.get_data_loader()

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.best_model_states = None
        self.model = Model(params).cuda()

        self.ce_loss = CrossEntropyLoss(
            label_smoothing=getattr(self.params, "label_smoothing", 0.0)
        ).cuda()
        self.criterion_cls = self.ce_loss

        self.coral_loss = CORAL().cuda()
        self.ae_loss = AELoss().cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.lr / 10
        )

        self.data_length = len(self.data_loader['train'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.params.epochs * self.data_length
        )

        self.num_classes = getattr(self.params, "num_classes", getattr(self.params, "num_of_classes", 5))

        self.ce_weight = getattr(self.params, "ce_weight", 1.0)
        self.lambda_coral = getattr(self.params, "lambda_coral", 0.5)
        self.lambda_ae = getattr(self.params, "lambda_ae", 0.5)

        self.lambda_elr = getattr(self.params, "lambda_elr", 0.5)
        self.elr_momentum = getattr(self.params, "elr_momentum", 0.9)
        self.elr_warmup_start = getattr(self.params, "elr_warmup_start", 0)
        self.elr_warmup_end = getattr(self.params, "elr_warmup_end", 10)
        self.elr_warmup_init = getattr(self.params, "elr_warmup_init", 0.0)

        self.lambda_fourier = getattr(self.params, "lambda_fourier", 0.5)
        self.fourier_momentum = getattr(self.params, "fourier_momentum", 0.9)

        self.lambda_infomax = getattr(self.params, "lambda_infomax", 0.1)
        self.forward_free_loss = ForwardFreeRobustCELoss(
            num_classes=self.num_classes,
            lambda_infomax=self.lambda_infomax
        ).cuda()

        train_len = getattr(self.data_loader['train'].dataset, "__len__", lambda: None)()
        if train_len is None:
            train_len = self.data_length * getattr(self.params, "batch_size", 32)

        seq_len = getattr(self.params, "seq_len", 20)

        self.elr_buf = ELRBuffer(
            num_samples=train_len * seq_len,
            num_classes=self.num_classes,
            momentum=self.elr_momentum,
            device="cuda"
        )
        self.fourier_buf = FourierBuffer(
            num_samples=train_len,
            momentum=self.fourier_momentum,
            device="cuda"
        )

        self._fallback_idx_ptr = 0

        log_dir = os.path.join(self.params.log_dir, self.params.exp_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        print("[TB] log_dir =", log_dir)

        if not hasattr(self.params, "labels_are_noisy"):
            self.params.labels_are_noisy = True

        print(self.model)
        print("[LOSS] Using Forward-FREE Robust CE (InfoMax, NO T) + ELR + Fourier-ELR.")
        print(f"[NOISE] labels_are_noisy={self.params.labels_are_noisy}")
        print(f"[InfoMax] lambda={self.lambda_infomax}")

    @staticmethod
    def _linear_warmup(epoch: int, start: int, end: int, init_val: float, final_val: float) -> float:
        if end <= start:
            return final_val
        if epoch < start:
            return init_val
        if epoch >= end:
            return final_val
        r = (epoch - start) / float(end - start)
        return init_val * (1 - r) + final_val * r

    def _extract_batch(self, batch):
        if isinstance(batch, dict):
            x = batch.get('data')
            y = batch.get('label')
            z = batch.get('z', None)
            idx = batch.get('index', None)
            return x, y, z, idx

        if isinstance(batch, (list, tuple)):
            if len(batch) == 4:
                x, y, z, idx = batch
            elif len(batch) == 3:
                x, y, z = batch
                idx = None
            elif len(batch) == 2:
                x, y = batch
                z, idx = None, None
            else:
                raise ValueError("Unsupported batch structure.")
            return x, y, z, idx

        raise ValueError("Unsupported batch type.")

    def _expand_idx(self, idx, L, device):
        if idx is None:
            return None
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx, device=device)
        else:
            idx = idx.to(device)

        if idx.dim() == 2:
            return idx.reshape(-1)
        if idx.dim() == 1:
            if L > 1:
                idx = idx.unsqueeze(1).repeat(1, L)
            return idx.reshape(-1)

        raise ValueError(f"Unsupported idx shape: {idx.shape}")

    def _fallback_idx(self, N, device):
        start = self._fallback_idx_ptr
        idx = torch.arange(start, start + N, device=device) % self.elr_buf.P.size(0)
        self._fallback_idx_ptr = (start + N) % self.elr_buf.P.size(0)
        return idx

    def train(self):
        acc_best = 0.0
        f1_best = 0.0
        best_f1_epoch = 0

        try:
            for epoch in range(self.params.epochs):
                self.model.train()
                start_time = timer()
                losses = []

                elr_coef = self._linear_warmup(
                    epoch,
                    self.elr_warmup_start,
                    self.elr_warmup_end,
                    self.elr_warmup_init,
                    1.0,
                )

                for batch in tqdm(self.data_loader['train'], mininterval=10):
                    self.optimizer.zero_grad()

                    x, y, z, idx = self._extract_batch(batch)
                    x = x.cuda()
                    y = y.cuda()
                    z = z.cuda() if z is not None else None

                    pred, recon, mu = self.model(x)  
                    B, L, C = pred.shape
                    logits = pred

                    noisy_y = y

                    probs_detach = torch.softmax(logits.detach(), dim=-1)  
                    if idx is not None:
                        idx_vec = self._expand_idx(idx, L, logits.device)  
                    else:
                        idx_vec = self._fallback_idx(B * L, logits.device)

                    self.elr_buf.update(idx_vec, probs_detach.reshape(-1, C))

                    loss_cls, loss_ce_only, loss_mi_only = self.forward_free_loss(
                        logits,
                        noisy_y,
                        sample_weights=None
                    )

                    P_hist = self.elr_buf.get(idx_vec).reshape(B * L, C)
                    probs_now = torch.softmax(logits.reshape(-1, C), dim=-1)
                    loss_elr_raw = elr_regularizer(probs_now, P_hist, lambda_elr=self.lambda_elr)
                    loss_elr = elr_coef * loss_elr_raw

                    mu_f = torch.fft.rfft(mu, dim=1)
                    mu_f_mag = mu_f.abs()

                    seq_idx = idx_vec.reshape(B, L)[:, 0] if idx_vec is not None else self._fallback_idx(B, mu.device)
                    self.fourier_buf.update(seq_idx, mu_f_mag.detach())

                    P_hist_f = self.fourier_buf.get(seq_idx)
                    loss_fourier = fourier_elr_regularizer(mu_f_mag, P_hist_f, lambda_fourier=self.lambda_fourier)

                    loss_coral = self.coral_loss(mu, z) if z is not None else torch.tensor(0.0, device=logits.device)
                    loss_ae = self.ae_loss(x, recon)

                    loss = (
                        self.ce_weight * loss_cls
                        + loss_elr
                        + loss_fourier
                        + self.lambda_coral * loss_coral
                        + self.lambda_ae * loss_ae
                    )

                    loss.backward()
                    losses.append(loss.detach().cpu().item())

                    if getattr(self.params, "clip_value", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                    self.optimizer.step()
                    self.scheduler.step()

                optim_state = self.optimizer.state_dict()

                with torch.no_grad():
                    (
                        acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1,
                    ) = self.val_eval.get_accuracy(self.model)

                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, f1: {:.5f}, "
                    "LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60,
                    )
                )
                print(f"[warmup] elr_coef={elr_coef:.3f} (cls: ce={loss_ce_only.item():.4f}, mi={loss_mi_only.item():.4f})")
                print(cm)
                print(
                    "wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, "
                    "n3_f1: {:.5f}, rem_f1: {:.5f}".format(
                        wake_f1, n1_f1, n2_f1, n3_f1, rem_f1
                    )
                )

                self.writer.add_scalar("loss/train", float(np.mean(losses)), epoch + 1)
                self.writer.add_scalar("acc/val", float(acc), epoch + 1)
                self.writer.add_scalar("f1/val", float(f1), epoch + 1)

                if acc > acc_best:
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    f1_best = f1
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
                    print(
                        "Epoch {}: ACC increasing!! New acc: {:.5f}, f1: {:.5f}".format(
                            best_f1_epoch, acc_best, f1_best
                        )
                    )

            print("{} epoch get the best acc {:.5f} and f1 {:.5f}".format(best_f1_epoch, acc_best, f1_best))
            test_acc, test_f1 = self.test()
            return test_acc, test_f1

        finally:
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.close()

    def test(self):
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            (
                test_acc, test_f1, test_cm,
                test_wake_f1, test_n1_f1, test_n2_f1, test_n3_f1, test_rem_f1,
            ) = self.test_eval.get_accuracy(self.model)

        print("***************************Test results************************")
        print("Test Evaluation: acc: {:.5f}, f1: {:.5f}".format(test_acc, test_f1))
        print(test_cm)
        print(
            "wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, "
            "n3_f1: {:.5f}, rem_f1: {:.5f}".format(
                test_wake_f1, test_n1_f1, test_n2_f1, test_n3_f1, test_rem_f1,
            )
        )

        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)

        model_path = os.path.join(
            self.params.model_dir,
            "tacc_{:.5f}_tf1_{:.5f}.pth".format(test_acc, test_f1),
        )
        torch.save(self.best_model_states, model_path)
        print("the model is save in " + model_path)

        return test_acc, test_f1
