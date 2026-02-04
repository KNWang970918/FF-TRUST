import argparse
import os
import copy
import random
import numpy as np
import torch

from trainer import Trainer

datasets = [
    #'sleep-edfx',
    'HMC',
    #'ISRUC',
    #'SHHS1',
    #'P2018',
]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed = 0
    cuda_id = 0
    setup_seed(seed)
    torch.cuda.set_device(cuda_id)

    save_dir = "/cvhci/temp/knwang/SleepDG-main/Comparison Chart"
    os.makedirs(save_dir, exist_ok=True)

    noise_rates = [0.6]

    parser = argparse.ArgumentParser(description='FF-TRUST')

    parser.add_argument('--target_domains', type=str, default='sleep-edfx', help='target_domains')
    parser.add_argument('--train_on_target', action='store_true',
                        help='Use the same domain as both source (train/val) and target (test)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_of_classes', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1.2e-3, help='learning rate')
    parser.add_argument('--clip_value', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss')
    parser.add_argument('--datasets_dir', type=str, default='/cvhci/temp/knwang/SleepDG-main/datasets/Datasets')
    parser.add_argument('--model_dir', type=str, default='/cvhci/temp/knwang/SleepDG-main/checkpoints/ablation')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument("--log_dir", default="runs")
    parser.add_argument("--exp_name", default="exp5_HMC_SleepDG")
   
    parser.add_argument('--ce_weight', type=float, default=1.1)
    parser.add_argument('--lambda_coral', type=float, default=0.5)
    parser.add_argument('--lambda_ae', type=float, default=0.5)

    parser.add_argument('--lambda_infomax', type=float, default=0.2,
                        help='InfoMax regularization weight for Forward-Free loss')

    parser.add_argument('--lambda_elr', type=float, default=0.1)
    parser.add_argument('--elr_momentum', type=float, default=0.9)
    parser.add_argument('--seq_len', type=int, default=20)

    parser.add_argument('--lambda_fourier', type=float, default=0.1)
    parser.add_argument('--fourier_momentum', type=float, default=0.92)

    parser.add_argument('--elr_warmup_start', type=int, default=20)
    parser.add_argument('--elr_warmup_end', type=int, default=35)
    parser.add_argument('--elr_warmup_init', type=float, default=0.0)
    
    parser.add_argument('--noise_mode', type=str, default='asymmetric',
                        choices=['none', 'symmetric', 'asymmetric'])
    parser.add_argument('--noise_rate', type=float, default=0.6)
    parser.add_argument('--noise_seed', type=int, default=0)
    parser.add_argument('--noise_gamma', type=float, default=1.0,
                        help='asymmetric: 概率 ~ exp(-gamma * |i-j|)')
    
    base_args = parser.parse_args()

    results = {nr: {ds: {} for ds in datasets} for nr in noise_rates}

    for nr in noise_rates:
        print(f"\n============ Noise rate = {nr} ============\n")
        for dataset_name in datasets:
            setup_seed(seed)
            params = copy.deepcopy(base_args)
            params.target_domains = dataset_name
            params.noise_rate = nr

            print(
                f'Running dataset="{dataset_name}" | lr={params.lr} | '
                f'noise: mode={params.noise_mode}, rate={params.noise_rate}, '
                f'seed={params.noise_seed}, gamma={params.noise_gamma} | '
                f'ce_weight={params.ce_weight},lambda_coral={params.lambda_coral} | '
                f'lambda_ae={params.lambda_ae},lambda_infomax={params.lambda_infomax} | '
                f'lambda_elr={params.lambda_elr},elr_momentum={params.elr_momentum} | '
                f'lambda_fourier={params.lambda_fourier},fourier_momentum={params.fourier_momentum} | '
            )

            trainer = Trainer(params)
            model = trainer.model   
            num_total = sum(p.numel() for p in model.parameters())
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"[Model Params] Total: {num_total:,} | Trainable: {num_trainable:,}")
            test_acc, test_f1 = trainer.train()
            results[nr][dataset_name] = (test_acc, test_f1)

    avg_acc_sums = np.zeros(len(noise_rates), dtype=float)
    avg_f1_sums = np.zeros(len(noise_rates), dtype=float)
    num_datasets_run = 0

    for dataset_name in datasets:
        accs, f1s = [], []
        for nr in noise_rates:
            if results[nr][dataset_name]:
                acc, f1 = results[nr][dataset_name]
            else:
                acc, f1 = 0.0, 0.0
            accs.append(float(acc))
            f1s.append(float(f1))


        np.savetxt(os.path.join(save_dir, f'noise_sweep_{dataset_name}.csv'),
                   np.column_stack([noise_rates, accs, f1s]),
                   delimiter=',', header='noise_rate,test_acc,test_mf1', comments='')

        print(dataset_name, 'noise_rates:', noise_rates)
        print(dataset_name, 'accs:', accs)
        print(dataset_name, 'mf1s:', f1s)

        avg_acc_sums += np.array(accs)
        avg_f1_sums += np.array(f1s)
        num_datasets_run += 1

    if num_datasets_run > 0:
        avg_acc = (avg_acc_sums / num_datasets_run).tolist()
        avg_f1  = (avg_f1_sums / num_datasets_run).tolist()

        np.savetxt(os.path.join(save_dir, 'noise_sweep_average.csv'),
                   np.column_stack([noise_rates, avg_acc, avg_f1]),
                   delimiter=',', header='noise_rate,avg_test_acc,avg_test_mf1', comments='')

        print('AVERAGE noise_rates:', noise_rates)
        print('AVERAGE accs:', avg_acc)
        print('AVERAGE mf1s:', avg_f1)

if __name__ == '__main__':
    main()
