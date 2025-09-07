from typing import Optional

import torch
import torch.nn as nn


class CrossAttn2D(nn.Module):
    """
    Cross-Attention module for fusing text messages with 2D image features.

    This module enables cross-modal attention between text message tokens and
    spatial image features. Message tokens act as queries to attend to image
    features (keys/values), creating a fused representation that incorporates
    semantic information from the text into the spatial image representation.

    The fusion process:
    1. Project image features to attention dimension
    2. Cross-attention: message queries attend to image features
    3. Aggregate attended message features and broadcast to all image locations
    4. Add the broadcasted message information to image features
    5. Project back to original image channel dimension

    Args:
        ch (int): Number of input/output channels in the feature maps
        d_model (int): Dimension for attention computation (should match message encoder)
        nhead (int, optional): Number of attention heads. Defaults to 4.

    Input:
        fmap (torch.Tensor): Input feature maps of shape (batch_size, ch, height, width)
        msg_tokens (torch.Tensor): Message token embeddings of shape (batch_size, seq_len, d_model)
        msg_pad_mask (torch.Tensor, optional): Padding mask for messages (currently unused)

    Output:
        torch.Tensor: Message-conditioned feature maps of shape (batch_size, ch, height, width)

    Example:
        >>> cross_attn = CrossAttn2D(ch=256, d_model=512, nhead=8)
        >>> fmap = torch.randn(4, 256, 32, 32)    # Image features
        >>> msg = torch.randn(4, 20, 512)        # Message embeddings
        >>> output = cross_attn(fmap, msg)
        >>> print(output.shape)  # torch.Size([4, 256, 32, 32])

    Note:
        - d_model should match the message encoder's output dimension
        - The attention mechanism allows each message token to focus on relevant image regions
        - Final fusion broadcasts global message information to all spatial locations
    """

    def __init__(self, ch: int, d_model: int, nhead: int = 4):
        super().__init__()

        # Project input feature maps to attention dimension
        self.in_proj = nn.Conv2d(
            ch, d_model, 1
        )  # 1x1 conv: (B, ch, H, W) -> (B, d_model, H, W)

        # Multi-head cross-attention mechanism
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )  # Expects (B, L, D) format

        # Project back to original feature map channels
        self.out_proj = nn.Conv2d(
            d_model, ch, 1
        )  # 1x1 conv: (B, d_model, H, W) -> (B, ch, H, W)

        # Layer normalization for stable attention computation
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        fmap: torch.Tensor,
        msg_tokens: torch.Tensor,
        msg_pad_mask: Optional[torch.Tensor] = None,
    ):
        """
        Apply cross-attention between message tokens and image features.

        Args:
            fmap (torch.Tensor): Feature maps of shape (batch_size, ch, height, width)
            msg_tokens (torch.Tensor): Message embeddings of shape (batch_size, seq_len, d_model)
            msg_pad_mask (torch.Tensor, optional): Message padding mask (currently unused)

        Returns:
            torch.Tensor: Message-conditioned features of shape (batch_size, ch, height, width)
        """
        # fmap: (B, C, H, W), msg_tokens: (B, L, D)
        B, C, H, W = fmap.shape

        # Step 1: Project image features to attention dimension
        x = self.in_proj(fmap)  # (B, ch, H, W) -> (B, d_model, H, W)

        # Step 2: Reshape image features for attention computation
        # Flatten spatial dimensions to create sequence of image tokens
        x_tokens = x.flatten(2).transpose(1, 2)  # (B, d_model, HW) -> (B, HW, d_model)

        # Step 3: Set up cross-attention components
        # Queries: message tokens (what information to extract)
        Q = msg_tokens  # (B, seq_len, d_model)

        # Keys: normalized image tokens (where to look in the image)
        K = self.norm(x_tokens)  # (B, HW, d_model)

        # Values: raw image tokens (what information to extract)
        V = x_tokens  # (B, HW, d_model)

        # Step 4: Apply cross-attention
        # Each message token attends to relevant parts of the image
        fused, _ = self.attn(Q, K, V, key_padding_mask=None)  # (B, seq_len, d_model)

        # Step 5: Aggregate message information for broadcasting
        # Compute global message representation by averaging across sequence
        m = fused.mean(dim=1, keepdim=True)  # (B, seq_len, d_model) -> (B, 1, d_model)

        # Step 6: Broadcast message information to all image locations
        # Add the global message vector to every spatial location
        x_tokens = x_tokens + m.expand(-1, x_tokens.size(1), -1)  # (B, HW, d_model)

        # Step 7: Reshape back to spatial format
        x = x_tokens.transpose(1, 2).view(
            B, -1, H, W
        )  # (B, HW, d_model) -> (B, d_model, H, W)

        # Step 8: Project back to original channel dimension
        return self.out_proj(x)  # (B, d_model, H, W) -> (B, ch, H, W)
