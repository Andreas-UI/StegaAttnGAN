import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
from StegaAttnGAN import (
    StegaAttnGAN,
    StegaConfig,
    rand_tokens_with_pad,
    stego_losses,
    psnr,
    to_0_1,
    to_minus1_1,
    evaluate_step
)
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset
from collections import defaultdict
import random


@torch.no_grad()
def evaluate_and_visualize(
    model: StegaAttnGAN, images, cfg: StegaConfig, msg_len=None, n_vis=4
):
    model.eval()
    B = images.size(0)

    if msg_len is None:
        msg_len = cfg.max_msg_len
    tokens, token_pad_mask = rand_tokens_with_pad(
        B, msg_len, cfg.max_msg_len, cfg.vocab_size
    )
    tokens = tokens.to(images.device)
    token_pad_mask = token_pad_mask.to(images.device)
    tgt_len = cfg.max_msg_len

    # Forward
    stego = model.forward_generator(images, tokens, token_pad_mask)
    stego_noised = model.noise(stego)
    logits = model.forward_decoder(stego_noised, tgt_len)

    adv_real = model.disc(images)
    adv_fake = model.disc(stego)

    # Loss + PSNR
    losses = stego_losses(images, stego, logits, tokens[:, :tgt_len], adv_fake, adv_real)
    quality_psnr = psnr(images, stego)

    # Decode text
    decoded_tokens = torch.argmax(logits, dim=-1)

    # Visualization
    if n_vis > 0:
        n_vis = min(n_vis, B)
        fig, axes = plt.subplots(n_vis, 3, figsize=(9, 3 * n_vis))
        if n_vis == 1:
            axes = np.expand_dims(axes, axis=0)

        # Uncomment if want to show, and viceversa
        for i in range(n_vis):
            cover_img = to_0_1(images[i].cpu().permute(1, 2, 0)).clamp(0, 1)
            stego_img = to_0_1(stego[i].cpu().permute(1, 2, 0)).clamp(0, 1)
            residual = (
                torch.abs(stego_img - cover_img) * 5
            )  # amplify differences for visibility

            axes[i, 0].imshow(cover_img)
            axes[i, 0].set_title("Cover")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(stego_img)
            axes[i, 1].set_title("Stego")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(residual)
            axes[i, 2].set_title("Residual (x5)")
            axes[i, 2].axis("off")

            print(f"Sample {i + 1}:")
            print(f"  Original: {tokens[i, :tgt_len].cpu().tolist()}")
            print(f"  Decoded : {decoded_tokens[i, :tgt_len].cpu().tolist()}")

        plt.tight_layout()
        plt.show()

    return {"psnr": quality_psnr, **{k: v.item() for k, v in losses.items()}}


def load_model_from_checkpoint(checkpoint_path, cfg, device="cuda"):
    # Recreate model architecture
    model = StegaAttnGAN(cfg).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Loaded model from {checkpoint_path}")
    return model


device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = StegaConfig()
cfg.vocab_size = 128
cfg.max_msg_len = 3
model_ = load_model_from_checkpoint("stega_exports/checkpoint.pth", cfg, "cuda")
# batch_size = 512 * 2
batch_size = 32

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(to_minus1_1),
    ]
)

cifar10_val = torchvision.datasets.CIFAR10(
    root="./data", download=True, transform=transform, train=False
)


class ImageOnlyWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image  # discard label


testloader = torch.utils.data.DataLoader(
    ImageOnlyWrapper(ConcatDataset([cifar10_val])),
    batch_size=batch_size,
    # pin_memory=True,
    # num_workers=torch.get_num_threads() - 10,
    shuffle=True,
    # prefetch_factor=8,
    # persistent_workers=True,
)

for batch in testloader:
    batch = batch.to(device)
    log = evaluate_and_visualize(model_, batch, cfg, msg_len=3, n_vis=5)
    print(log)
    break  # remove break if you want more



# def evaluate_full_testset(model, testloader, cfg, msg_len=3, device="cuda"):
#     all_metrics = defaultdict(float)
#     total_samples = 0

#     with torch.no_grad():
#         for batch in testloader:
#             batch = batch.to(device)
#             log = evaluate_and_visualize(model, batch, cfg, 1, 0)  # use evaluate_step for metrics

#             for k, v in log.items():
#                 all_metrics[k] += v * batch.size(0)  # weight by sample count
#             total_samples += batch.size(0)

#     avg_metrics = {k: v / total_samples for k, v in all_metrics.items()}
#     return avg_metrics


# def run_multiple_evals(model, testloader, cfg, msg_len=3, n_runs=10, device="cuda"):
#     results = []

#     for run in range(n_runs):
#         # Different seed for each run
#         torch.manual_seed(run)
#         np.random.seed(run)
#         random.seed(run)

#         avg_metrics = evaluate_full_testset(model, testloader, cfg, msg_len, device)
#         results.append(avg_metrics)

#     # Aggregate mean and std
#     keys = results[0].keys()
#     final_avg = {k: np.mean([r[k] for r in results]) for k in keys}
#     final_std = {k: np.std([r[k] for r in results]) for k in keys}

#     print(f"\n📊 Results over {n_runs} runs (mean ± std):")
#     for k in keys:
#         print(f"{k}: {final_avg[k]:.4f} ± {final_std[k]:.4f}")

#     return final_avg, final_std


# # Usage
# final_avg, final_std = run_multiple_evals(
#     model_, testloader, cfg, msg_len=2, n_runs=10, device=device
# )
