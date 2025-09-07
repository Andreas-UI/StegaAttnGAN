import torch
import torch.nn as nn

from StegaAttnGAN.layers import ConvBlock
from StegaAttnGAN.transformer import PositionalEncoding


class Decoder(nn.Module):
    """
    Image-to-Text Decoder using CNN feature extraction and Transformer decoding.

    This decoder extracts spatial features from images using a CNN backbone,
    then uses a Transformer decoder to generate text sequences by cross-attending
    to the image features. The architecture combines computer vision and NLP
    components for image-to-text tasks like captioning or steganographic decoding.

    Architecture:
    1. CNN Backbone: Extracts spatial features (downsamples to 8x8 grid)
    2. Feature Projection: Projects CNN features to Transformer dimension
    3. Transformer Decoder: Cross-attends from text queries to image features
    4. Token Head: Projects to vocabulary space for text generation

    Args:
        vocab_size (int): Size of the vocabulary for text generation
        d_model (int, optional): Transformer model dimension. Defaults to 256.
        base (int, optional): Base number of CNN channels. Defaults to 64.
        max_len (int, optional): Maximum text sequence length. Defaults to 256.

    Input:
        img (torch.Tensor): Input images of shape (batch_size, 3, height, width)
        tgt_len (int): Target sequence length to generate

    Output:
        torch.Tensor: Logits for text tokens of shape (batch_size, tgt_len, vocab_size)

    Example:
        >>> decoder = Decoder(vocab_size=5000, d_model=256, base=64, max_len=128)
        >>> img = torch.randn(4, 3, 32, 32)  # CIFAR-sized images
        >>> logits = decoder(img, tgt_len=20)
        >>> print(logits.shape)  # torch.Size([4, 20, 5000])

    Note:
        - Designed for CIFAR-scale images (32x32) with final feature grid of 8x8
        - Uses cross-attention mechanism where text queries attend to image features
        - Target sequence is initialized with zeros (could use learned start tokens)
    """

    def __init__(
        self, vocab_size: int, d_model: int = 256, base: int = 64, max_len: int = 256
    ):
        super().__init__()
        # CNN Feature Extractor (backbone)
        # Progressively downsamples input image to extract hierarchical features
        self.backbone = nn.Sequential(
            # Initial feature extraction
            ConvBlock(3, base),  # (B, 3, H, W) -> (B, 64, H, W)
            ConvBlock(base, base),  # (B, 64, H, W) -> (B, 64, H, W)
            # First downsampling: 32x32 -> 16x16
            nn.Conv2d(base, base * 2, 3, 2, 1),  # Stride=2 for downsampling
            ConvBlock(base * 2, base * 2),  # (B, 128, 16, 16) -> (B, 128, 16, 16)
            # Second downsampling: 16x16 -> 8x8
            nn.Conv2d(base * 2, base * 4, 3, 2, 1),  # Stride=2 for downsampling
            ConvBlock(base * 4, base * 4),  # (B, 256, 8, 8) -> (B, 256, 8, 8)
        )

        # Project CNN features to Transformer dimension
        self.proj = nn.Conv2d(
            base * 4, d_model, 1
        )  # 1x1 conv: (B, 256, 8, 8) -> (B, d_model, 8, 8)

        # Transformer Decoder
        # Single decoder layer configuration
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,  # Model dimension
            nhead=4,  # Number of attention heads
            dim_feedforward=d_model * 4,  # FFN inner dimension
            batch_first=True,  # Batch dimension first
        )

        # Stack 3 decoder layers
        self.dec = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # Output projection to vocabulary
        self.token_head = nn.Linear(d_model, vocab_size)  # (B, L, D) -> (B, L, V)

        # Positional encodings
        self.pos_tgt = PositionalEncoding(d_model, max_len)  # For target sequences
        self.pos_mem = PositionalEncoding(d_model, 64)  # For 8x8=64 spatial locations

        # Store maximum sequence length
        self.max_len = max_len

    def forward(self, img: torch.Tensor, tgt_len: int):
        """
        Forward pass through the image-to-text decoder.

        Args:
            img (torch.Tensor): Input images of shape (batch_size, 3, height, width)
            tgt_len (int): Length of target sequence to generate

        Returns:
            torch.Tensor: Token logits of shape (batch_size, tgt_len, vocab_size)
        """
        B = img.size(0)  # Batch size
        # Step 1: Extract spatial features using CNN backbone
        feat = self.backbone(img)  # (B, 3, H, W) -> (B, base*4, 8, 8)

        # Step 2: Project to Transformer dimension
        feat = self.proj(feat)  # (B, base*4, 8, 8) -> (B, d_model, 8, 8)

        # Step 3: Reshape spatial features for Transformer
        # Flatten spatial dimensions and transpose for sequence format
        mem = feat.flatten(2).permute(0, 2, 1)  # (B, d_model, 64) -> (B, 64, d_model)

        # Step 4: Add positional encoding to image features (memory)
        mem = self.pos_mem(mem)  # (B, 64, d_model)

        # Step 5: Initialize target sequence
        # Start with zero embeddings (could use learned start tokens)
        tgt = torch.zeros(
            B, tgt_len, mem.size(-1), device=img.device
        )  # (B, tgt_len, d_model)

        # Step 6: Add positional encoding to target sequence
        tgt = self.pos_tgt(tgt)  # (B, tgt_len, d_model)

        # Step 7: Cross-attention decoding
        # Target queries attend to image memory features
        out = self.dec(tgt=tgt, memory=mem)  # (B, tgt_len, d_model)

        # Step 8: Project to vocabulary space
        logits = self.token_head(out)  # (B, tgt_len, vocab_size)
        return logits
