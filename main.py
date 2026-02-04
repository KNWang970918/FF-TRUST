import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
from timeit import default_timer as timer
from utils import *
import random
from datasets.dataset import LoadDataset
from trainer import Trainer

# datasets = {
#             'sleep-edfx': 0,
#             'HMC': 1,
#             'ISRUC': 2,
#             'SHHS1': 3,
#             'P2018': 4,
#         }


def main():
    parser = argparse.ArgumentParser(description='SleepDG')
    parser.add_argument('--target_domains', type=str, default='sleep-edfx', help='target_domains')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--clip_value', type=float, default=0.5, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--loss_function', type=str, default='WeightedCE', help='CrossEntropyLoss | WeightedCE | FocalLoss')
    parser.add_argument('--datasets_dir', type=str, default='/cvhci/temp/knwang/SleepDG-main/datasets/Datasets', help='datasets_dir')
    parser.add_argument('--model_dir', type=str, default='/cvhci/temp/knwang/SleepDG-main/checkpoints', help='model_dir')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='label_smoothing')
    parser.add_argument('--lambda_coral', type=float, default=0.1)
    parser.add_argument('--lambda_ae', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--ce_weight', type=float, default=0.5, help='weight for CE loss (paper: 0.5)')
    parser.add_argument('--lambda_coral', type=float, default=0.5, help='weight for CORAL loss (paper: 0.5)')
    parser.add_argument('--lambda_ae', type=float, default=0.5, help='weight for AE loss (paper: 0.5)')
    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)

    torch.cuda.set_device(params.cuda)

    trainer = Trainer(params)
    trainer.train()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()