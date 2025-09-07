import torch
from StegaAttnGAN.loss import psnr, stego_losses
from StegaAttnGAN.util import rand_tokens_with_pad
from StegaAttnGAN.model import StegaAttnGAN, StegaConfig

@torch.no_grad()
def evaluate_step(model: StegaAttnGAN, images, cfg: StegaConfig, msg_len=None):
    model.eval()
    B = images.size(0)

    # Random message if not provided
    if msg_len is None:
        msg_len = cfg.max_msg_len
    tokens, token_pad_mask = rand_tokens_with_pad(
        B, msg_len, cfg.max_msg_len, cfg.vocab_size
    )
    tokens = tokens.to(images.device)
    token_pad_mask = token_pad_mask.to(images.device)

    cover = images
    tgt_len = msg_len

    # Generator forward
    stego = model.forward_generator(cover, tokens, token_pad_mask)

    # Apply robustness noise
    stego_noised = model.noise(stego)

    # Decode
    logits = model.forward_decoder(stego_noised, tgt_len)

    # Discriminator
    adv_real = model.disc(cover)
    adv_fake = model.disc(stego)

    # Losses
    losses = stego_losses(
        cover,
        stego,
        logits,
        tokens[:, :tgt_len],
        adv_fake,
        adv_real,
        pad_mask=token_pad_mask[:, :tgt_len] if token_pad_mask is not None else None,
    )

    # PSNR
    quality_psnr = psnr(cover, stego)

    return {"psnr": quality_psnr, **{k: v.item() for k, v in losses.items()}}


def training_step(
    model: StegaAttnGAN, batch, opt_g, opt_d, cfg: StegaConfig, phase="both"
):
    model.train()
    cover, tokens, token_pad_mask, tgt_len = batch

    # forward generator
    stego = model.forward_generator(cover, tokens, token_pad_mask)
    stego_noised = model.noise(stego)
    logits = model.forward_decoder(stego_noised, tgt_len)

    adv_real = model.disc(cover)
    adv_fake = model.disc(stego.detach())

    # loss weights based on phase
    if phase == "text":
        img_w, txt_w, gan_w = 0.1, 1.0, 0.0  # mostly text
    elif phase == "image":
        img_w, txt_w, gan_w = 1.0, 0.2, 1.0  # mostly image & GAN
    else:
        img_w, txt_w, gan_w = 1.0, 1.0, 1.0  # balanced

    losses = stego_losses(
        cover,
        stego,
        logits,
        tokens[:, :tgt_len],
        adv_fake,
        adv_real,
        img_loss_w=img_w,
        txt_loss_w=txt_w,
        gan_loss_w=gan_w,
        pad_mask=token_pad_mask[:, :tgt_len] if token_pad_mask is not None else None,
    )

    # update D
    opt_d.zero_grad(set_to_none=True)
    losses["gan_d"].backward()
    opt_d.step()

    # update G
    adv_fake_for_g = model.disc(stego)
    losses_g = stego_losses(
        cover,
        stego,
        logits,
        tokens[:, :tgt_len],
        adv_fake_for_g,
        adv_real,
        img_loss_w=img_w,
        txt_loss_w=txt_w,
        gan_loss_w=gan_w,
        pad_mask=token_pad_mask[:, :tgt_len] if token_pad_mask is not None else None,
    )
    opt_g.zero_grad(set_to_none=True)
    losses_g["gen_total"].backward()
    opt_g.step()

    with torch.no_grad():
        quality_psnr = psnr(cover, stego)

    log = {"psnr": quality_psnr, **{k: v.item() for k, v in losses_g.items()}}
    return log
