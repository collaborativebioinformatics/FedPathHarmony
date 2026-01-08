# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License").

"""
NVFlare Job recipe for FedAvg with site-specific Prostate dataset
"""

import argparse
from nets.models import DenseNet, ClientModel1  # or UNet if using segmentation
from utils.dataset import Camelyon17

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

# from nets.models_factory import make_densenet


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=5, help="Number of federated sites/clients")
    parser.add_argument("--num_rounds", type=int, default=200, help="Number of FL rounds")
    parser.add_argument("--epochs", type=int, default=2, help="Local training epochs per round")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds

    # Initial model for FedAvg
    initial_model = DenseNet(input_shape=[3, 96, 96], num_classes=2)  # adapt input_shape and num_classes
    # initial_model = ClientModel1(backbone='densenet', do_norm=False)

    sites = [1, 2, 3, 4, 5]
    # Define the FedAvg recipe
    recipe = FedAvgRecipe(
        name="camelyon-fedavg",
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=initial_model,   # <-- call the factory
        train_script="harmo_flare/client.py",  # your client code for site-specific dataset
        train_args=f"--epochs {args.epochs} --batch_size {args.batch_size}",
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
