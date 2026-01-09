import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .nets.models import BeerLaNet, DenseNet
from viz_utils import *

to_tensor = T.ToTensor()

def main():
    # same structure as before
    PKL_PATH = "/home/ubuntu/Cross_biobank_data_access/camelyon17/data.pkl"
    SPLIT = "train"          # "train" or "test"
    BASE_PATH = "camelyon17/data"   # join here if p is relative
    OUT_DIR = "sample"
    K = 5                    # samples per site

    # BeerLaNet params
    RANK_R = 3               # 6 D channels
    N_ITER = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(PKL_PATH, allow_pickle=True)

    # model = BeerLaNet(r=RANK_R, c=3, learn_S_init=True, calc_tau=True).to(device)

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(nrows=3, ncols=5, dpi=300)

    for site in [1, 2, 3, 4, 5]:

        model = DenseNet(do_norm=True)
        model.to(device)
        model.eval()

        key = f"hospital{site}"
        if key not in data:
            print(f"[WARN] missing key in pkl: {key}")
            continue

        paths, _labels = data[key][SPLIT]
        paths = list(paths)


        chosen = random.sample(paths, k=min(K, len(paths)))
        src = os.path.join(BASE_PATH, chosen[0]) 
        img = Image.open(src).convert("RGB")
        ax[0, site-1].imshow(img)


        for i in range(2):
            D = run_beerlanet_get_D(img, model, n_iter=N_ITER, device=device)
            D = (D - D.min()) / (D.max() - D.min())
            ax[i+1, site-1].imshow(D.detach().cpu().permute(1, 2, 0))

            model = load_params(model)


        ax[0, site-1].get_xaxis().set_ticks([])
        ax[0, site-1].get_yaxis().set_ticks([])
        ax[1, site-1].get_xaxis().set_ticks([])
        ax[1, site-1].get_yaxis().set_ticks([])
        ax[2, site-1].get_xaxis().set_ticks([])
        ax[2, site-1].get_yaxis().set_ticks([])
        ax[0, site-1].set_title(f'Client {site}')


    ax[0, 0].set_ylabel('Original')
    ax[1, 0].set_ylabel('Harmonized (Untrained)')
    ax[2, 0].set_ylabel('Harmonized')
    plt.tight_layout()
    plt.savefig('harmonization.png')

        # for i, p in enumerate(chosen):
        #     src = p if os.path.isabs(p) or BASE_PATH is None else os.path.join(BASE_PATH, p)
        #     if not os.path.exists(src):
        #         print(f"[site{site}] missing: {src}")
        #         continue

        #     try:
        #         img = Image.open(src).convert("RGB")
        #     except Exception as e:
        #         print(f"[site{site}] failed to open {src}: {e}")
        #         continue

        #     try:
        #         D = run_beerlanet_get_D(img, model, n_iter=N_ITER, device=device)  # [6,H,W]
        #         stitched = stitch_original_plus_D(img, D)                          # [7 panels]
        #     except Exception as e:
        #         print(f"[site{site}] BeerLaNet failed on {src}: {e}")
        #         continue

        #     out_path = os.path.join(site_out, f"{i}.png")
        #     stitched.save(out_path)
        #     print(f"[site{site}] saved -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()