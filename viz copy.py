import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from nets.models import BeerLaNet, DenseNet

to_tensor = T.ToTensor()

def load_params(model):
    ckpt = torch.load("/tmp/nvflare/simulation/camelyon-fedharmo/server/simulate_job/app_server/best_FL_global_model.pt", map_location="cpu", weights_only=True)
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
    model = DenseNet(do_norm=True)
    model = load_params(model)
    model.to(device)
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)

    for site in [1, 2, 3, 4, 5]:
        key = f"hospital{site}"
        if key not in data:
            print(f"[WARN] missing key in pkl: {key}")
            continue

        paths, _labels = data[key][SPLIT]
        paths = list(paths)

        site_out = os.path.join(OUT_DIR, f"site{site}")
        os.makedirs(site_out, exist_ok=True)

        chosen = random.sample(paths, k=min(K, len(paths)))

        for i, p in enumerate(chosen):
            src = p if os.path.isabs(p) or BASE_PATH is None else os.path.join(BASE_PATH, p)
            if not os.path.exists(src):
                print(f"[site{site}] missing: {src}")
                continue

            try:
                img = Image.open(src).convert("RGB")
            except Exception as e:
                print(f"[site{site}] failed to open {src}: {e}")
                continue

            try:
                D = run_beerlanet_get_D(img, model, n_iter=N_ITER, device=device)  # [6,H,W]
                stitched = stitch_original_plus_D(img, D)                          # [7 panels]
            except Exception as e:
                print(f"[site{site}] BeerLaNet failed on {src}: {e}")
                continue

            out_path = os.path.join(site_out, f"{i}.png")
            stitched.save(out_path)
            print(f"[site{site}] saved -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()