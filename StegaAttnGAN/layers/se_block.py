import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for channel attention mechanism.

    The SE block recalibrates feature maps by explicitly modeling channel
    interdependencies. It performs global average pooling to squeeze spatial
    information, then uses two FC layers to learn channel attention weights,
    and finally applies these weights to the input features.

    Paper: "Squeeze-and-Excitation Networks" (https://arxiv.org/abs/1709.01507)

    Args:
        ch (int): Number of input/output channels
        r (int, optional): Reduction ratio for the bottleneck layer. Defaults to 8.
            Controls the capacity and computational cost of the SE block.

    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, ch, height, width)

    Output:
        torch.Tensor: Output tensor of shape (batch_size, ch, height, width)
        Input features recalibrated by learned channel attention weights.

    Example:
        >>> se_block = SEBlock(ch=256, r=16)
        >>> x = torch.randn(4, 256, 32, 32)
        >>> output = se_block(x)
        >>> print(output.shape)  # torch.Size([4, 256, 32, 32])
    """

    def __init__(self, ch, r=8):
        super().__init__()

        # Sequential block implementing the SE mechanism:
        # 1. Squeeze: Global average pooling (H×W -> 1×1)
        # 2. Excitation: Two FC layers with ReLU and Sigmoid activation
        self.fc = nn.Sequential(
            # Squeeze: Global average pooling to compress spatial dimensions
            nn.AdaptiveAvgPool2d(1),  # (B, ch, H, W) -> (B, ch, 1, 1)
            # Excitation part 1: Dimensionality reduction
            nn.Conv2d(ch, ch // r, 1),  # (B, ch, 1, 1) -> (B, ch//r, 1, 1)
            nn.ReLU(inplace=True),
            # Excitation part 2: Dimensionality restoration + attention weights
            nn.Conv2d(ch // r, ch, 1),  # (B, ch//r, 1, 1) -> (B, ch, 1, 1)
            nn.Sigmoid(),  # Output attention weights in range [0, 1]
        )

    def forward(self, x):
        """
        Forward pass through the SE block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ch, height, width)

        Returns:
            torch.Tensor: Channel-recalibrated features of shape (batch_size, ch, height, width)
        """
        # Generate channel attention weights
        w = self.fc(x)  # Shape: (batch_size, ch, 1, 1)

        # Apply channel-wise multiplication (broadcasting across spatial dimensions)
        return x * w  # Element-wise multiplication recalibrates each channel
