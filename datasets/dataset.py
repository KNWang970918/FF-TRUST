import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.util import to_tensor


def _build_linear_distance_matrix(num_classes: int) -> np.ndarray:
    idx = np.arange(num_classes)
    return np.abs(idx[:, None] - idx[None, :]).astype(float)


def _distance_row_to_probs(dist_row: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    d = np.asarray(dist_row, dtype=float).copy()
    d[np.isclose(d, 0.0)] = np.inf
    logits = -gamma * d
    logits[np.isinf(logits)] = -np.inf
    finite = np.isfinite(logits)
    if not np.any(finite):
        w = np.zeros_like(d)
        mask = ~np.isinf(d)
        w[mask] = 1.0 / mask.sum()
        return w
    m = np.max(logits[finite])
    ex = np.exp(logits - m)
    ex[~np.isfinite(ex)] = 0.0
    s = ex.sum()
    if s <= 0:
        w = np.zeros_like(d)
        mask = ~np.isinf(d)
        w[mask] = 1.0 / mask.sum()
        return w
    return ex / s


def inject_label_noise_array(
    y_clean,
    num_classes: int = 5,
    noise_rate: float = 0.2,
    mode: str = 'symmetric',      
    rng=None,
    gamma: float = 1.0,           
):
    rng = np.random.default_rng(rng)
    y_clean = np.asarray(y_clean).copy()
    y_noisy = y_clean.copy()
    L = y_clean.shape[0]

    if mode == 'none' or noise_rate <= 0:
        return y_noisy

    if mode == 'symmetric':
        flip = rng.random(L) < noise_rate
        idxs = np.where(flip)[0]
        for i in idxs:
            true = int(y_clean[i])
            cand = list(range(num_classes))
            cand.remove(true)
            y_noisy[i] = rng.choice(cand)
        return y_noisy

    if mode == 'asymmetric':
        D = _build_linear_distance_matrix(num_classes)
        row_probs = [_distance_row_to_probs(D[i], gamma=gamma) for i in range(num_classes)]
        flip = rng.random(L) < noise_rate
        idxs = np.where(flip)[0]
        for i in idxs:
            true = int(y_clean[i])
            p = row_probs[true]
            y_noisy[i] = rng.choice(np.arange(num_classes), p=p)
        return y_noisy

    raise ValueError("noise_mode must be 'none', 'symmetric', or 'asymmetric'")

class CustomDataset(Dataset):
    def __init__(
        self,
        seqs_labels_path_pair,
        apply_noise: bool = False,      
        num_classes: int = 5,
        noise_mode: str = 'none',       
        noise_rate: float = 0.0,
        rng_seed: int = 0,
        gamma: float = 1.0,             
    ):
        super().__init__()
        self.seqs_labels_path_pair = seqs_labels_path_pair

        self.apply_noise = bool(apply_noise)
        self.num_classes = int(num_classes)
        self.noise_mode = str(noise_mode)
        self.noise_rate = float(noise_rate)
        self.rng_seed = int(rng_seed)
        self.gamma = float(gamma)

        self._noisy_label_cache = {}
        if self.apply_noise and self.noise_mode != 'none' and self.noise_rate > 0:
            for idx, (_, label_path, _) in enumerate(self.seqs_labels_path_pair):
                y_clean = np.load(label_path)
                y_noisy = inject_label_noise_array(
                    y_clean,
                    num_classes=self.num_classes,
                    noise_rate=self.noise_rate,
                    mode=self.noise_mode,
                    rng=(self.rng_seed + idx),
                    gamma=self.gamma,
                )
                self._noisy_label_cache[idx] = y_noisy

    def __len__(self):
        return len(self.seqs_labels_path_pair)

    def __getitem__(self, idx):
        seq_path, label_path, subject_id = self.seqs_labels_path_pair[idx]
        seq = np.load(seq_path)
        seq_eeg = seq[:, :1, :]
        seq_eog = seq[:, 1:2, :]
        seq = np.concatenate((seq_eeg, seq_eog), axis=1) 

        if self.apply_noise and self.noise_mode != 'none' and self.noise_rate > 0:
            label = self._noisy_label_cache[idx]
        else:
            label = np.load(label_path)  
        return seq, label, subject_id, idx

    def collate(self, batch):
        x_seq   = np.array([x[0] for x in batch]) 
        y_label = np.array([x[1] for x in batch])  
        z_label = np.array([x[2] for x in batch])  
        idx_arr = np.array([x[3] for x in batch])  
        return (
            to_tensor(x_seq),
            to_tensor(y_label).long(),
            to_tensor(z_label).long(),
            to_tensor(idx_arr).long(),
        )

    def collate_eval(self, batch):
        x_seq   = np.array([x[0] for x in batch])  
        y_label = np.array([x[1] for x in batch]) 
        z_label = np.array([x[2] for x in batch])  
        return (
            to_tensor(x_seq),
            to_tensor(y_label).long(),
            to_tensor(z_label).long(),
        )

class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets = {
            'sleep-edfx': 0,
            'HMC': 1,
            'ISRUC': 2,
            'SHHS1': 3,
            'P2018': 4,
        }

        td = self.params.target_domains
        if isinstance(td, str):
            self.target_names = [td]
        elif isinstance(td, (list, tuple, set)):
            self.target_names = list(td)
        else:
            raise ValueError("target_domains should be str or list-like")

        base = self.params.datasets_dir
        self.targets_dirs = [f'{base}/{k}' for k in self.datasets.keys() if k in self.target_names]
        self.source_dirs  = [f'{base}/{k}' for k in self.datasets.keys() if k not in self.target_names]
        print("targets_dirs:", self.targets_dirs)
        print("source_dirs :", self.source_dirs)

        self.num_classes = getattr(self.params, 'num_of_classes', 5)
        self.noise_mode = getattr(self.params, 'noise_mode', 'none')     
        self.noise_rate = getattr(self.params, 'noise_rate', 0.0)
        self.noise_seed = getattr(self.params, 'noise_seed', 0)
        self.noise_gamma = getattr(self.params, 'noise_gamma', 1.0)

    def get_data_loader(self):
        if getattr(self.params, 'train_on_target', False):
            all_pairs, _ = self._load_path(self.targets_dirs, 0)
            random.shuffle(all_pairs)
            n = len(all_pairs)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            train_pairs = all_pairs[:n_train]
            val_pairs = all_pairs[n_train:n_train + n_val]
            test_pairs = all_pairs[n_train + n_val:]
            print(f"[train_on_target] total={n}, train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

            train_set = CustomDataset(
                train_pairs,
                apply_noise=(self.noise_mode != 'none' and self.noise_rate > 0),
                num_classes=self.num_classes,
                noise_mode=self.noise_mode,
                noise_rate=self.noise_rate,
                rng_seed=self.noise_seed,
                gamma=self.noise_gamma,
            )
            val_set = CustomDataset(val_pairs)    
            test_set = CustomDataset(test_pairs)   

        else:
            source_domains, subject_id = self._load_path(self.source_dirs, 0)
            target_domains, _ = self._load_path(self.targets_dirs, subject_id)
            train_pairs, val_pairs = self._split_dataset(source_domains)
            print(f"[DG] train={len(train_pairs)}, val={len(val_pairs)}, test={len(target_domains)}")

            train_set = CustomDataset(
                train_pairs,
                apply_noise=(self.noise_mode != 'none' and self.noise_rate > 0),
                num_classes=self.num_classes,
                noise_mode=self.noise_mode,
                noise_rate=self.noise_rate,
                rng_seed=self.noise_seed,
                gamma=self.noise_gamma,
            )
            val_set = CustomDataset(val_pairs)            
            test_set = CustomDataset(target_domains)     

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,        
                shuffle=True,          
                num_workers=self.params.num_workers,
                drop_last=False,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate_eval,     
                shuffle=False,
                num_workers=self.params.num_workers,
                drop_last=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate_eval,    
                shuffle=False,
                num_workers=self.params.num_workers,
                drop_last=False,
            ),
        }
        return data_loader, subject_id

    def _load_path(self, domains_dirs, subject_id_start: int):
        domains = []
        sid = subject_id_start
        for dataset in domains_dirs:
            seq_root = os.path.join(dataset, 'seq')
            labels_root = os.path.join(dataset, 'labels')
            if not (os.path.isdir(seq_root) and os.path.isdir(labels_root)):
                continue
            seq_dirs = sorted(os.listdir(seq_root))
            labels_dirs = sorted(os.listdir(labels_root))
            for seq_dir, labels_dir in zip(seq_dirs, labels_dirs):
                seq_dir_full = os.path.join(seq_root, seq_dir)
                labels_dir_full = os.path.join(labels_root, labels_dir)
                seq_names = sorted(os.listdir(seq_dir_full))
                labels_names = sorted(os.listdir(labels_dir_full))
                for seq_name, labels_name in zip(seq_names, labels_names):
                    domains.append((
                        os.path.join(seq_dir_full, seq_name),
                        os.path.join(labels_dir_full, labels_name),
                        sid
                    ))
            sid += 1
        return domains, sid

    def _split_dataset(self, source_domains):
        random.shuffle(source_domains)
        split_num = int(len(source_domains) * 0.8)
        train_pairs = source_domains[:split_num]
        val_pairs = source_domains[split_num:]
        return train_pairs, val_pairs


if __name__ == '__main__':
    import argparse

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='SleepDG')
    parser.add_argument('--target_domains', type=str, nargs='+', default=['SHHS1'], help='target domain names')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--datasets_dir', type=str, default='--datasets_dir')

    parser.add_argument('--num_of_classes', type=int, default=5)
    parser.add_argument('--noise_mode', type=str, default='none',
                        choices=['none', 'symmetric', 'asymmetric'])
    parser.add_argument('--noise_rate', type=float, default=0.0)
    parser.add_argument('--noise_seed', type=int, default=0)
    parser.add_argument('--noise_gamma', type=float, default=1.0,
                        help='asymmetric: 概率 ~ exp(-gamma * |i-j|)')

    params = parser.parse_args()
    setup_seed(params.seed)

    loader, _ = LoadDataset(params).get_data_loader()
    print("Train batches:", len(loader['train']))
    print("Val batches:", len(loader['val']))
    print("Test batches:", len(loader['test']))
