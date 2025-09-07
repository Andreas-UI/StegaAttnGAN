from typing import Optional

import torch
import torch.nn as nn

from StegaAttnGAN.transformer import PositionalEncoding


class MessageEncoder(nn.Module):
    """
    Transformer-based encoder for text message processing in steganographic systems.

    Encodes text messages (token sequences) into rich contextual representations
    using the Transformer encoder architecture. The encoded representations can
    then be used for steganographic embedding into images or other modalities.

    Architecture:
    Token IDs -> Embedding -> Positional Encoding -> Transformer Encoder -> Layer Norm

    The encoder uses self-attention to capture dependencies between tokens and
    produces contextualized embeddings that preserve semantic meaning while
    being suitable for steganographic operations.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens)
        d_model (int, optional): Model dimension for embeddings and attention. Defaults to 256.
        nhead (int, optional): Number of attention heads in each layer. Defaults to 4.
        num_layers (int, optional): Number of Transformer encoder layers. Defaults to 2.

    Input:
        tokens (torch.Tensor): Token IDs of shape (batch_size, sequence_length)
        key_padding_mask (torch.Tensor, optional): Mask for padded tokens of shape
                                                  (batch_size, sequence_length).
                                                  True indicates padded positions.

    Output:
        torch.Tensor: Encoded message representations of shape (batch_size, sequence_length, d_model)

    Example:
        >>> encoder = MessageEncoder(vocab_size=5000, d_model=256, nhead=8, num_layers=3)
        >>> tokens = torch.randint(0, 5000, (4, 20))  # Batch of 4 sequences, length 20
        >>> mask = torch.zeros(4, 20, dtype=torch.bool)  # No padding
        >>> encoded = encoder(tokens, mask)
        >>> print(encoded.shape)  # torch.Size([4, 20, 256])

    Note:
        - Uses batch_first=True convention for easier handling
        - Layer normalization applied at the end for stable training
        - Supports variable-length sequences via key_padding_mask
        - d_model should be divisible by nhead for multi-head attention
    """

    def __init__(
        self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 2
    ):
        super().__init__()

        # Token embedding layer: maps discrete tokens to continuous vectors
        self.embed = nn.Embedding(vocab_size, d_model)  # (vocab_size,) -> (d_model,)

        # Single Transformer encoder layer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # Model/embedding dimension
            nhead=nhead,  # Number of attention heads
            dim_feedforward=d_model * 4,  # Inner FFN dimension (common 4x scaling)
            batch_first=True,  # Batch dimension comes first: (B, L, D)
        )

        # Stack multiple encoder layers
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Positional encoding to inject sequence order information
        self.pos = PositionalEncoding(d_model)

        # Final layer normalization for training stability
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Encode token sequences into contextualized representations.

        Args:
            tokens (torch.Tensor): Input token IDs of shape (batch_size, sequence_length)
            key_padding_mask (torch.Tensor, optional): Padding mask of shape
                                                      (batch_size, sequence_length).
                                                      True for padded positions to ignore.

        Returns:
            torch.Tensor: Encoded representations of shape (batch_size, sequence_length, d_model)
        """
        # Step 1: Convert token IDs to embeddings
        x = self.embed(tokens)  # (B, L) -> (B, L, D)

        # Step 2: Add positional encoding to preserve sequence order
        x = self.pos(x)  # (B, L, D) -> (B, L, D) [adds position info]

        # Step 3: Apply Transformer encoder layers with self-attention
        # key_padding_mask prevents attention to padded tokens
        x = self.enc(x, src_key_padding_mask=key_padding_mask)  # (B, L, D) -> (B, L, D)

        # Step 4: Apply final layer normalization
        return self.norm(x)  # (B, L, D)
