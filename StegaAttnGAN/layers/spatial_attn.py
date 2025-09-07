import torch
import torch.nn as nn


class SpatialAttn(nn.Module):
    """
    Spatial Attention Module for spatial feature recalibration.

    This module computes spatial attention weights by aggregating channel information
    using average and max pooling, then learning spatial attention maps through
    a convolutional layer. It helps the network focus on important spatial locations.

    Commonly used in CBAM (Convolutional Block Attention Module) and similar
    attention mechanisms for computer vision tasks.

    Args:
        ch (int): Number of input channels (not used in computation but kept for API consistency)

    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

    Output:
        torch.Tensor: Spatially recalibrated features of shape (batch_size, channels, height, width)

    Example:
        >>> spatial_attn = SpatialAttn(ch=256)
        >>> x = torch.randn(4, 256, 32, 32)
        >>> output = spatial_attn(x)
        >>> print(output.shape)  # torch.Size([4, 256, 32, 32])
    """

    def __init__(self, ch):
        super().__init__()

        # 7x7 convolution to generate spatial attention map from 2-channel input
        # Input: concatenated avg and max pooled features (2 channels)
        # Output: single channel attention map
        self.conv = nn.Conv2d(
            2, 1, 7, padding=3
        )  # kernel_size=7, padding=3 preserves spatial dims

    def forward(self, x):
        """
        Forward pass through the spatial attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Spatially attended features of shape (batch_size, channels, height, width)
        """

        # Aggregate channel information using statistical operations
        # Average pooling across channels - captures average activation strength
        avg = torch.mean(x, dim=1, keepdim=True)  # (B, C, H, W) -> (B, 1, H, W)

        # Max pooling across channels - captures strongest activation
        mx, _ = torch.max(x, dim=1, keepdim=True)  # (B, C, H, W) -> (B, 1, H, W)

        # Concatenate avg and max pooled features along channel dimension
        s = torch.cat([avg, mx], dim=1)  # (B, 1, H, W) + (B, 1, H, W) -> (B, 2, H, W)

        # Generate spatial attention weights using 7x7 convolution + sigmoid
        w = torch.sigmoid(self.conv(s))  # (B, 2, H, W) -> (B, 1, H, W)

        # Apply spatial attention weights (broadcasting across channel dimension)
        return x * w  # (B, C, H, W) * (B, 1, H, W) -> (B, C, H, W)
