# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License").

"""
NVFlare Job recipe for FedAvg with site-specific Prostate dataset
"""

import argparse
from nets.models import DenseNet, ClientModel  # or UNet if using segmentation
from utils.dataset import Camelyon17

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking
import torch

# from nets.models_factory import make_densenet


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=5, help="Number of federated sites/clients")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of FL rounds")
    parser.add_argument("--epochs", type=int, default=0, help="Local training epochs per round")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--fl_type", type=str, default="fedavg")
    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds

    if args.fl_type == "fedavg":
        do_norm = False
        exp_name = "test-camelyon-fedavg"
        file_path_name = "camelyon-fedavg_20"
    elif args.fl_type == "harmo":
        do_norm = True
        exp_name = "test-camelyon-fedharmo"
        file_path_name = "camelyon-fedharmo_20"


    initial_model = DenseNet(input_shape=[3, 96, 96], num_classes=2, do_norm=do_norm)  # adapt input_shape and num_classes
    model_path = f"/tmp/nvflare/simulation/{file_path_name}/server/simulate_job/app_server/FL_global_model.pt"

    ckpt = torch.load(model_path, map_location="cpu")["model"]    
    initial_model.load_state_dict(ckpt, strict=False)


    # Define the FedAvg recipe
    recipe = FedAvgRecipe(
        name=exp_name,
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=initial_model,   # <-- call the factory
        train_script="harmo_flare/test_client.py",  # your client code for site-specific dataset
        train_args=f"--batch_size {args.batch_size} --fl_type {args.fl_type} --model_path {model_path}",
        # train_args=[f"--site {site} --epochs {args.epochs} --batch_size {args.batch_size}" for site in sites]
    )
    

    # Optional: TensorBoard tracking
    # add_experiment_tracking(recipe, tracking_type="tensorboard")

    # Simulated environment (for local testing)
    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)
    

    print()
    print("Job Status:", run.get_status())
    print("Results can be found at:", run.get_result())
    print()


if __name__ == "__main__":
    main()
