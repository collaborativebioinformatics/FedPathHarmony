# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import argparse
import os
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from nets.models import DenseNet  # or your UNet/other model
from utils.dataset import Camelyon17  # your dataset class

# NVFlare imports
import nvflare.client as flare

import json
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve, auc



# def evaluate(net, data_loader, device, do_norm=False):
#     print("evaluating")
#     net.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in data_loader:
#             images, labels = data[0].to(device), data[1].to(device)
#             if do_norm:
#                 outputs, _, _ = net(images)
#             else:
#                 outputs,_, _ = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     print(f"Accuracy: {accuracy:.2f}%")
#     return accuracy

def evaluate(net, data_loader, device, do_norm=False, results_path=None):
    print("Evaluating...")
    net.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            
            # Forward pass
            if do_norm:
                outputs, _, _ = net(images)
            else:
                outputs, _, _ = net(images)
            
            probs = torch.softmax(outputs, dim=1)  # class probabilities

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    num_samples = len(all_labels)

    # Predicted classes
    preds = torch.argmax(all_probs, dim=1)

    # Metrics
    accuracy = 100.0 * (preds == all_labels).sum().item() / len(all_labels)
    f1 = f1_score(all_labels, preds, average="weighted")

    # AUROC
    auroc = roc_auc_score(all_labels, all_probs[:,1])
   

    # Confusion matrix
    cm = confusion_matrix(all_labels, preds)

    
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:,1])
    roc_auc = auc(fpr, tpr)
    

    print(f"Accuracy: {accuracy:.2f}%, F1: {f1:.3f}, AUROC: {auroc:.3f}")
    print("Confusion Matrix:")
    print(cm)

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "auroc": auroc,
        "num_samples": num_samples
        # "confusion_matrix": cm.tolist(),  # convert to list for JSON dumping
        # "fpr": fpr.tolist() if fpr is not None else None,
        # "tpr": tpr.tolist() if tpr is not None else None,
        # "roc_auc_curve": roc_auc
    }

    # Dump to JSON
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fl_type", type=str, default="fedavg")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    batch_size = args.batch_size

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
    model_path = args.model_path

    ckpt = torch.load(model_path, map_location="cpu")["model"]    
    model.load_state_dict(ckpt, strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = Camelyon17(site=site_name, split="test", transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    

    print("FLARE RUNNING:", flare.is_running())

    results_root = f"/home/ubuntu/Cross_biobank_data_access/results/{args.fl_type}"
    results_path = os.path.join(results_root, f"{client_name}_20.json")
    print(results_path)


    while flare.is_running():
        # Receive global model from server
        print("-----------------------------")
        input_model = flare.receive()
        print(f"site={client_name}, current_round={input_model.current_round}")

        # Load received model
        model.load_state_dict(ckpt)
        model.to(device)

        # Evaluate
        accuracy = evaluate(model, test_loader, device, do_norm, results_path)

        print(f"Finished training for {client_name}")

        # Construct output FL model and send to server
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"accuracy": accuracy},
        )
        print(f"site={client_name}, sending model to server.")

        
        flare.send(output_model)


if __name__ == "__main__":
    main()
