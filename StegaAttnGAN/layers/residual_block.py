import torch.nn as nn
from StegaAttnGAN.layers import ConvBlock


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection for deep network training.

    Implements the core building block from ResNet architecture that uses
    skip connections to enable training of very deep networks by mitigating
    the vanishing gradient problem. The block applies two ConvBlock operations
    and adds the input (identity mapping) to the output.

    Architecture: Input -> ConvBlock -> ConvBlock -> Add Input -> Output

    Paper: "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385)

    Args:
        ch (int): Number of input/output channels (must be the same for residual connection)

    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, ch, height, width)

    Output:
        torch.Tensor: Output tensor of shape (batch_size, ch, height, width)
        The spatial dimensions remain unchanged due to the ConvBlock default parameters.

    Example:
        >>> residual_block = ResidualBlock(ch=256)
        >>> x = torch.randn(4, 256, 32, 32)
        >>> output = residual_block(x)
        >>> print(output.shape)  # torch.Size([4, 256, 32, 32])

    Note:
        This assumes ConvBlock maintains spatial dimensions (default k=3, s=1, p=1).
        The number of input and output channels must be identical for the skip connection.
    """

    def __init__(self, ch):
        super().__init__()

        # Sequential block containing two ConvBlock layers
        # Each ConvBlock: Conv2d -> BatchNorm2d -> LeakyReLU
        # Both blocks have the same number of input/output channels
        self.block = nn.Sequential(
            ConvBlock(ch, ch),  # First ConvBlock: (B, ch, H, W) -> (B, ch, H, W)
            ConvBlock(ch, ch),  # Second ConvBlock: (B, ch, H, W) -> (B, ch, H, W)
        )

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ch, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ch, height, width)
        """
        # Apply the sequential ConvBlock operations
        residual = self.block(x)

        # Add skip connection: F(x) + x
        # This enables gradient flow and helps training deeper networks
        return x + residual
