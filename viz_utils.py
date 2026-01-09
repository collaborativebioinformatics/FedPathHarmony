import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from nets.models import BeerLaNet, DenseNet


def load_params(model, model_type='harm'):
    if model_type == 'harm':
        ckpt = torch.load("/tmp/nvflare/simulation/camelyon-fedharmo/server/simulate_job/app_server/best_FL_global_model.pt", map_location="cpu", weights_only=True)
    elif model_type == 'fedavg':
        ckpt = torch.load("/tmp/nvflare/simulation/camelyon-fedavg/server/simulate_job/app_server/best_FL_global_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt['model'])
    return model

@torch.no_grad()
def run_beerlanet_get_D(img_rgb: Image.Image, model: BeerLaNet, n_iter: int, device: str) -> torch.Tensor:
    """
    Returns D for a single image: [r,H,W]
    """
    X = to_tensor(img_rgb).unsqueeze(0).to(device)  # [1,3,H,W]
    _, D = model(X)            # D: [1,r,H,W]
    # return D[0]                                     # [r,H,W]
    return D.squeeze(0)

def stitch_original_plus_D(img_rgb: Image.Image, D: torch.Tensor) -> Image.Image:
    """
    img_rgb: PIL RGB (W,H)
    D: torch [r,H,W] (single image)
    Returns a stitched PIL RGB: [Original | D1 | ... | Dr]
    """
    W, H = img_rgb.size
    r, H_d, W_d = D.shape
    assert (H_d, W_d) == (H, W), "D spatial size must match image"

    # Convert D channels to grayscale PILs, then to RGB for consistent stitching
    D_np = D.detach().cpu().numpy()  # [r,H,W]
    d_imgs = [_d_channel_to_pil_gray(D_np[i]).convert("RGB") for i in range(r)]

    panels = [img_rgb] + d_imgs  # total 1+r panels
    out = Image.new("RGB", (W * (1 + r), H))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += W
    return out

def stitch_original_plus_harmon(img_rgb: Image.Image, D: torch.Tensor) -> Image.Image:
    """
    img_rgb: PIL RGB (W,H)
    D: torch [r,H,W] (single image)
    Returns a stitched PIL RGB: [Original | D1 | ... | Dr]
    """
    W, H = img_rgb.size
    r, H_d, W_d = D.shape
    assert (H_d, W_d) == (H, W), "D spatial size must match image"

    # Convert D channels to grayscale PILs, then to RGB for consistent stitching
    D_np = D.detach().cpu().numpy()  # [r,H,W]

    panels = [img_rgb] + D_np  # total 1+r panels
    out = Image.new("RGB", (W * 2, H))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += W
    return out



def _d_channel_to_pil_gray(d_2d: np.ndarray) -> Image.Image:
    """
    d_2d: [H,W] float
    Convert to 0..255 using per-channel min-max for visibility.
    """
    d_min = float(d_2d.min())
    d_max = float(d_2d.max())
    if d_max - d_min < 1e-12:
        out = np.zeros_like(d_2d, dtype=np.uint8)
    else:
        out = ((d_2d - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="L")
