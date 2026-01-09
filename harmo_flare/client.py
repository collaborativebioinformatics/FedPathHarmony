# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import argparse
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from nets.models import DenseNet  # or your UNet/other model
from utils.dataset import Camelyon17  # your dataset class

# NVFlare imports
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter


def evaluate(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _, _ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fl_type", type=str, default="fedavg")
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    if args.fl_type == "fedavg":
        do_norm = False
    elif args.fl_type == "harmo":
        do_norm = True

    # Initialize NVFlare client
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    site_name = int(client_name[-1])

    # Initialize model
    model = DenseNet(input_shape=[3, 96, 96], num_classes=2, do_norm=do_norm)  # adapt input_shape if needed
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Load dataset
    from torch.utils.data import random_split

    full_train_dataset = Camelyon17(site=site_name, split="train", transform=ToTensor())
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    print(train_size)
    print("0000000000000000000000000000000000000000")
    # train_dataset = Camelyon17(site=site_name, split="train", transform=ToTensor())
    # val_dataset = Camelyon17(site=site_name, split="val", transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    

    # Optional metrics tracking
    summary_writer = SummaryWriter()

    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        print(f"site={client_name}, current_round={input_model.current_round}")

        # Load received model
        model.load_state_dict(input_model.params)
        model.to(device)

        # Evaluate
        accuracy = evaluate(model, val_loader, device)

        # Local training
        steps = epochs * len(train_loader)
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                outputs, _, _ = model(images)
                cost = loss_fn(outputs, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.item()
                if i % 200 == 199:
                    avg_loss = running_loss / 200
                    print(f"[{epoch+1}, {i+1}] loss: {avg_loss:.4f}")
                    # log metrics
                    global_step = input_model.current_round * steps + epoch * len(train_loader) + i
                    summary_writer.add_scalar(tag="loss", scalar=avg_loss, global_step=global_step)
                    running_loss = 0.0
            

        print(f"Finished training for {client_name}")

        # Construct output FL model and send to server
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site={client_name}, sending model to server.")
        flare.send(output_model)


if __name__ == "__main__":
    main()
