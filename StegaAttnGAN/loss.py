import torch
import torch.nn.functional as F
import math


def stego_losses(
    cover,
    stego,
    logits,
    target_tokens,
    adv_fake,
    adv_real,
    img_loss_w=1.0,
    txt_loss_w=1.0,
    gan_loss_w=1.0,
    pad_mask=None,
):
    # image reconstruction loss
    img_loss = F.mse_loss(stego, cover)

    # text reconstruction loss
    if pad_mask is not None:
        mask = ~pad_mask.view(-1)  # keep non-PAD
        txt_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1))[mask], target_tokens.view(-1)[mask]
        )
    else:
        txt_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target_tokens.view(-1)
        )

    # GAN losses
    gan_d = 0.5 * (F.relu(1.0 - adv_real).mean() + F.relu(1.0 + adv_fake).mean())
    gan_g = -adv_fake.mean()

    # weighted generator total
    gen_total = img_loss_w * img_loss + txt_loss_w * txt_loss + gan_loss_w * gan_g

    return {
        "img_loss": img_loss * img_loss_w,
        "txt_loss": txt_loss * txt_loss_w,
        "gan_d": gan_d * gan_loss_w,
        "gan_g": gan_g * gan_loss_w,
        "gen_total": gen_total,
    }


@torch.no_grad()
def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    if mse == 0:
        return 99.0
    return 10 * math.log10(4.0 / mse)  # since range is [-1,1] -> peak-to-peak 2
