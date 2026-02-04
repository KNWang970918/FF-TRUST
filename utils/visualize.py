import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import Model
from tqdm import tqdm
import numpy as np
import torch
from datasets.dataset import LoadDataset
from evaluator import Evaluator
from torch.nn import CrossEntropyLoss
from timeit import default_timer as timer
import os
from sklearn import manifold
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FastICA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import pandas as pd
# sns.set(style="darkgrid")
OUTDIR = 'viz_out'
os.makedirs('viz_out', exist_ok=True)
label2color = {
    0: 'yellow',
    1: 'red',
    2: 'blue',
    3: 'pink',
    4: 'green',
}
label2class = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM',
}
font1 = {'family': 'Times New Roman'}
matplotlib.rc("font", **font1)
class Visualization(object):
    def __init__(self, params):
        self.params = params
        self.data_loader, subject_id = LoadDataset(params).get_data_loader()

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = Model(params).cuda()
        self.model.load_state_dict(torch.load(self.params.model_path, map_location='cpu'))
        print(self.model)

    def visualize(self):
        # ts = manifold.TSNE(n_components=2, random_state=42)
        ts = umap.UMAP(random_state=42)
        self.model.eval()
        feats_list = []
        labels_list = []
        i = 0
        for x, y, z in tqdm(self.data_loader['train']):
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            pred, recon, mu = self.model(x)
            # print(seq_features.shape)
            feats = mu.view(-1, 512)
            feats_list.append(feats.detach().cpu().numpy())
            labels = y.view(-1)
            labels_list.append(labels.detach().cpu().numpy())
            i += 1
            if i > 10:
                break
        feats_all = numpy.concatenate(feats_list, axis=0)
        labels_all = numpy.concatenate(labels_list, axis=0)


        feats_ts = ts.fit_transform(feats_all)
        print(feats_ts.shape)

        self.draw(feats_ts, labels_all)


    def draw(self, feats_ts, labels_all):
        xs = [[], [], [], [], []]
        ys = [[], [], [], [], []]
        for i, label in enumerate(labels_all):
            xs[label].append(feats_ts[i, 0])
            ys[label].append(feats_ts[i, 1])

        Wake = plt.scatter(x=xs[0], y=ys[0], c='yellow', alpha=1, marker='.', label='Wake')
        N1 = plt.scatter(x=xs[1], y=ys[1], c='red', alpha=1, marker='.', label='N1')
        N2 = plt.scatter(x=xs[2], y=ys[2], c='blue', alpha=1, marker='.', label='N2')
        N3 = plt.scatter(x=xs[3], y=ys[3], c='pink', alpha=1, marker='.', label='N3')
        REM = plt.scatter(x=xs[4], y=ys[4], c='green', alpha=1, marker='.', label='REM')
        plt.xticks([], fontsize=20)
        plt.yticks([], fontsize=20)
        plt.legend(fontsize=12)
        #plt.show()
        plt.savefig(os.path.join(OUTDIR, "visualize1.pdf"), bbox_inches='tight', pad_inches=0.0)
        plt.close()
        print("[viz] saved:", os.path.abspath(os.path.join(OUTDIR, "visualize1.pdf")))

    def visualize_correlation(self):
        ts = manifold.Isomap(n_components=1)
        # ts = PCA(n_components=1, random_state=42)
        self.model.eval()
        seqs_list = [[], [], [], [], []]
        # domains_list = []
        for x, y, z in tqdm(self.data_loader['train']):
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            pred, recon, mu = self.model(x)
            bz = z.shape[0]
            for bi in range(bz):
                dom = int(z[bi].item())
                seqs_list[dom].append(mu[bi].detach().cpu().numpy())
        for x, y, z in tqdm(self.data_loader['test']):
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            pred, recon, mu = self.model(x)
            bz = z.shape[0]
            for bi in range(bz):
                dom = int(z[bi].item()) 
                seqs_list[dom].append(mu[bi].detach().cpu().numpy())
        means_list = []
        for domain_id in range(5):
            arr = np.array(seqs_list[domain_id])
            means_list.append(np.zeros((20,512), dtype=np.float32) if arr.size == 0 else np.mean(arr, axis=0))

        means_array = np.concatenate(means_list, axis=0)
        print(means_array.shape)
        means_ts = ts.fit_transform(means_array)
        print(means_ts.shape)
        means_ts = means_ts.reshape(5, 20)
        self.draw_cor(means_ts)

    def draw_cor(self, means_ts):
        order  = [4, 0, 1, 2, 3]  # SleepEDFx, HMC, ISRUC, SHHS, P2018
        names  = ['SleepEDFx', 'HMC', 'ISRUC', 'SHHS', 'P2018']
        plt.figure()
        for idx, name in zip(order, names):
            plt.plot(means_ts[idx], marker='.', alpha=1, label=name)
        plt.legend(fontsize=12)
        plt.ylim(-7, 7)
        #plt.show()
        plt.savefig(os.path.join(OUTDIR, "visualize2.pdf"), bbox_inches='tight', pad_inches=0.0)
        plt.close()
        print("[viz] saved:", os.path.abspath(os.path.join(OUTDIR, "visualize2.pdf")))

    def visualize_cor_seaborn(self):
        # ts = manifold.Isomap(n_components=1)
        ts = PCA(n_components=1, random_state=42)
        self.model.eval()
        seqs_list = []
        domains_list = []
        n = 0
        for x, y, z in tqdm(self.data_loader['train']):
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            pred, recon, mu = self.model(x)
            bz = z.shape[0]
            seqs_list.append(mu.detach().cpu().numpy())
            domains_list.append(z.detach().cpu().numpy())
            n += 1
            if n > 100:
                break
        n = 0
        for x, y, z in tqdm(self.data_loader['test']):
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            pred, recon, mu = self.model(x)
            bz = z.shape[0]
            seqs_list.append(mu.detach().cpu().numpy())
            domains_list.append(z.detach().cpu().numpy())
            n += 1
            if n > 100:
                break
        seqs_array = np.concatenate(seqs_list, axis=0)
        domains_array = np.concatenate(domains_list, axis=0)
        print(seqs_array.shape, domains_array.shape)
        seqs_array = seqs_array.reshape(-1, 512)
        seqs_ts = ts.fit_transform(seqs_array)
        seqs_ts = seqs_ts.reshape(-1, 20)
        print(seqs_ts.shape)

        # print(means_ts.shape)
        # means_ts = means_ts.reshape(5, 20)
        self.draw_cor_seaborn(seqs_ts, domains_array)
        self.draw_c(seqs_ts, domains_array)
    def draw_c(self, seqs_ts, domains_array):
        order  = [4, 0, 1, 2, 3]
        id2paper  = {4:'SleepEDFx', 0:'HMC', 1:'ISRUC', 2:'SHHS', 3:'P2018'}
        domains_array = domains_array.astype(int)
        T = seqs_ts.shape[1]
        x = np.arange(T)
        plt.figure()
        for dom_id in order:
            idx = np.where(domains_array == dom_id)[0]
            if idx.size == 0:
                continue
            curves = seqs_ts[idx]            # [N_dom, 20]
            mean   = curves.mean(axis=0)     # [20]
            std    = curves.std(axis=0)      # [20]
            plt.plot(x, mean, linewidth=1.5, label=id2paper[dom_id])
            plt.fill_between(x, mean-std, mean+std, alpha=0.25)

        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([]); plt.yticks([])
        plt.legend(fontsize=12)
        plt.savefig(os.path.join(OUTDIR, "visualize3.pdf"), bbox_inches='tight', pad_inches=0.0)
        plt.close()
        print("[viz] saved:", os.path.abspath(os.path.join(OUTDIR, "visualize3.pdf")))
    
    def draw_cor_seaborn(self, seqs_ts, domains_array):
        order  = [4, 0, 1, 2, 3]
        names  = ['SleepEDFx', 'HMC', 'ISRUC', 'SHHS', 'P2018']
        id2paper = {4:'SleepEDFx', 0:'HMC', 1:'ISRUC', 2:'SHHS', 3:'P2018'}
        domains_array = domains_array.astype(int)
        data_list = []
        for i in range(seqs_ts.shape[0]):
            for j in range(20):
                data_list.append({
                    'feature': seqs_ts[i][j],
                    'time': j,
                    'Domains': id2paper[(domains_array[i])]
                })
        df = pd.DataFrame(data_list)
        sns.lineplot(x="time", y="feature",hue="Domains", style="Domains", data=df)
        # plt.xticks([], fontsize=20)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([], fontsize=20)
        plt.yticks([], fontsize=20)
        plt.ylim(-1.6, 1)
        plt.legend(fontsize=12)
        # plt.show()
        plt.savefig(os.path.join(OUTDIR, "visualize4.pdf"), bbox_inches='tight', pad_inches=0.0)
        plt.close()
        print("[viz] saved:", os.path.abspath(os.path.join(OUTDIR, "visualize4.pdf")))
from types import SimpleNamespace

if __name__ == "__main__":
    params = SimpleNamespace(
        # === 路径 ===
        model_path="/cvhci/temp/knwang/SleepDG-main/--model_dir/tacc_0.37856_tf1_0.13748.pth",
        datasets_dir="/cvhci/temp/knwang/SleepDG-main/datasets/Datasets",

        # === 域名要与 dataset.py 的键完全一致 ===
        source_domains=["HMC", "ISRUC", "SHHS1", "P2018"],  # 这个其实 dataset.py 没用到，但留着无妨
        target_domains=["sleep-edfx"],                      # 注意是小写+连字符

        # === DataLoader ===
        batch_size=16,
        num_workers=4,
        shuffle=False,
        drop_last=False,

        # === 模型/数据形状 ===
        in_channels=2,
        seq_len=20,
        sample_length=3000,
        n_classes=5,
        num_of_classes=5,          # 一些模块用这个名字，两个都给上更稳
        dropout=0.1,               # ← 缺的就是它
        loss_function="CrossEntropyLoss",  # 若模型里有用到

        # === 可视化输出目录 ===
        viz_out=OUTDIR,
    )

    vis = Visualization(params)
    with torch.no_grad():
        vis.visualize()               # -> viz_out/visualize1.pdf
        vis.visualize_correlation()   # -> viz_out/visualize2.pdf
        vis.visualize_cor_seaborn()   # -> viz_out/visualize3和4.pdf