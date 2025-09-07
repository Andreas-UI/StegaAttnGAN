import torch.nn as nn
from StegaAttnGAN.architecture import UNetGen
from StegaAttnGAN.layers import Decoder, Discriminator, NoiseLayer
from StegaAttnGAN.transformer import MessageEncoder


class StegaConfig:
    vocab_size: int = 128
    d_model: int = 96
    nhead: int = 3
    gen_base: int = 32
    dec_base: int = 32
    max_msg_len: int = 8


class StegaAttnGAN(nn.Module):
    def __init__(self, cfg: StegaConfig):
        super().__init__()
        self.msg_enc = MessageEncoder(cfg.vocab_size, cfg.d_model, cfg.nhead)
        self.gen = UNetGen(
            in_ch=3, base=cfg.gen_base, d_model=cfg.d_model, nhead=cfg.nhead
        )
        self.dec = Decoder(
            cfg.vocab_size,
            d_model=cfg.d_model,
            base=cfg.dec_base,
            max_len=cfg.max_msg_len,
        )
        self.disc = Discriminator()
        self.noise = NoiseLayer()

    def forward_generator(self, img, tokens, key_padding_mask=None):
        msg_tokens = self.msg_enc(tokens, key_padding_mask)  # (B, L, D)
        stego = self.gen(img, msg_tokens)
        return stego

    def forward_decoder(self, img_noised, tgt_len):
        logits = self.dec(img_noised, tgt_len)
        return logits
