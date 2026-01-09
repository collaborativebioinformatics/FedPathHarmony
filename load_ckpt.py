import torch

ckpt = torch.load("/tmp/nvflare/simulation/camelyon-fedavg/server/simulate_job/app_server/best_FL_global_model.pt", map_location="cpu")

# for key, value in ckpt.items():
#     print (key, value.keys())


print(ckpt["train_conf"]["train"])

