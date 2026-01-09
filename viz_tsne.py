import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from nets.models import BeerLaNet, DenseNet
from utils.dataset import Camelyon17  # your dataset class
from viz_utils import *



def get_embeddings(dst='viz', device='cuda:0', model_type='harm'):
    model = DenseNet(do_norm=model_type=='harm')
    model = load_params(model, model_type=model_type)
    model.to(device)
    model.eval()

    all_embs = torch.empty(0, 1024)
    all_labels = []
    site_names = []
    for site in [1, 2, 3, 4, 5]:
        test_dataset = Camelyon17(site=site, split="test", transform=ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

        # site_embs = torch.empty(0, 1024)
        # site_labels = []
        for i, (X, y) in enumerate(test_loader):
            X = X.to(device)

            pred, harm, proj = model(X)
            proj = proj.detach().cpu()
            all_embs = torch.cat([all_embs, proj], dim=0)
            all_labels += y.tolist()
            site_names += [site]*32

            if i > num_batch: break

    print(f'Calculating TSNE for site {site}')
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter_without_progress=300)
    proj = tsne.fit_transform(all_embs)

    df = pd.DataFrame({
        'site': site_names,
        'labels': all_labels,
        'proj-one': proj[:, 0],
        'proj-two': proj[:, 1]
    })

    df.to_csv(os.path.join(dst, f'embs_{model_type}.csv'))

def cat_df(src, dst):
    df1 = pd.read_csv(os.path.join(src, 'embs_site1.csv'))
    df2 = pd.read_csv(os.path.join(src, 'embs_site2.csv'))
    df3 = pd.read_csv(os.path.join(src, 'embs_site2.csv'))
    df4 = pd.read_csv(os.path.join(src, 'embs_site3.csv'))
    df5 = pd.read_csv(os.path.join(src, 'embs_site4.csv'))

    df_cat = pd.concat([df1, df2, df3, df4, df5])
    df_cat.to_csv(dst, index=False)

def plot_embs(ax, df_path):
    df = pd.read_csv(df_path)

    colors = ['blue', 'red', 'orange', 'purple', 'teal']
    shapes = ['o', 'v', 's', '*', 'X']

    for site in [1, 2, 3, 4, 5]:
        df_site = df[df['site'] == site]
        ax.scatter(df_site['proj-one'], df_site['proj-two'], color=colors[site-1], marker=shapes[site-1], label=f'Client {site}')


if __name__ == '__main__':
# same structure as before
    PKL_PATH = "/home/ubuntu/Cross_biobank_data_access/camelyon17/data.pkl"
    SPLIT = "train"          # "train" or "test"
    BASE_PATH = "camelyon17/data"   # join here if p is relative
    OUT_DIR = "sample"
    K = 5                    # samples per site

    # BeerLaNet params
    RANK_R = 3               # 6 D channels
    N_ITER = 10
    num_batch = 5

    # get_embeddings(model_type='harm')
    # cat_df('viz', 'viz/cat_fedavg.csv')
    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(12, 7))   
    plot_embs(ax[0], 'viz/embs_fedavg.csv')
    plot_embs(ax[1], 'viz/embs_harm.csv')
    ax[0].set_title("FedAvg")
    ax[1].set_title('FedPathHarmony')
    ax[0].get_xaxis().set_ticks([])
    ax[0].get_yaxis().set_ticks([])
    ax[1].get_xaxis().set_ticks([])
    ax[1].get_yaxis().set_ticks([])
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('viz/tsne.png')


    
