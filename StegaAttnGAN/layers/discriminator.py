import torch.nn as nn
from StegaAttnGAN.layers import ConvBlock


class Discriminator(nn.Module):
    """
    Convolutional Discriminator for GAN-based image classification.

    A CNN-based discriminator that classifies images as real or fake in adversarial
    training. Uses progressive downsampling through convolutional layers followed
    by global average pooling and a linear classifier. Designed for binary
    classification (real/fake) in GAN frameworks.

    Architecture:
    Input -> ConvBlock -> Conv+LeakyReLU -> Conv+LeakyReLU -> Conv+LeakyReLU
          -> GlobalAvgPool -> Linear -> Output

    The network progressively reduces spatial dimensions while increasing channel depth,
    culminating in a single scalar output representing the real/fake classification score.

    Args:
        in_ch (int, optional): Number of input channels. Defaults to 3 (RGB images).
        base (int, optional): Base number of channels for feature maps. Defaults to 64.

    Input:
        x (torch.Tensor): Input images of shape (batch_size, in_ch, height, width)

    Output:
        torch.Tensor: Classification scores of shape (batch_size, 1)
        Higher values typically indicate "real", lower values indicate "fake"

    Example:
        >>> discriminator = Discriminator(in_ch=3, base=64)
        >>> x = torch.randn(4, 3, 32, 32)  # CIFAR-sized images
        >>> scores = discriminator(x)
        >>> print(scores.shape)  # torch.Size([4, 1])

    Note:
        - First ConvBlock uses norm=False (common practice in GAN discriminators)
        - Uses LeakyReLU activations for gradient flow in adversarial training
        - Global average pooling makes the network adaptable to different input sizes
    """

    def __init__(self, in_ch=3, base=64):
        super().__init__()

        # Convolutional feature extractor
        self.net = nn.Sequential(
            # Initial feature extraction (no batch norm for discriminator stability)
            ConvBlock(in_ch, base, norm=False),  # (B, in_ch, H, W) -> (B, base, H, W)
            # First downsampling layer: reduce spatial dimensions by 2x
            nn.Conv2d(
                base, base * 2, 4, 2, 1
            ),  # (B, base, H, W) -> (B, base*2, H/2, W/2)
            nn.LeakyReLU(0.1, inplace=True),  # Activation with negative slope
            # Second downsampling layer: reduce spatial dimensions by 2x
            nn.Conv2d(
                base * 2, base * 4, 4, 2, 1
            ),  # (B, base*2, H/2, W/2) -> (B, base*4, H/4, W/4)
            nn.LeakyReLU(0.1, inplace=True),  # Activation with negative slope
            # Feature refinement layer: same spatial size, more channels
            nn.Conv2d(
                base * 4, base * 4, 3, 1, 1
            ),  # (B, base*4, H/4, W/4) -> (B, base*4, H/4, W/4)
            nn.LeakyReLU(0.1, inplace=True),  # Activation with negative slope
            # Global average pooling: spatial dimensions -> 1x1
            nn.AdaptiveAvgPool2d(1),  # (B, base*4, H/4, W/4) -> (B, base*4, 1, 1)
        )

        # Classification head: maps features to single score
        self.head = nn.Linear(base * 4, 1)  # (B, base*4) -> (B, 1)

    def forward(self, x):
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, in_ch, height, width)

        Returns:
            torch.Tensor: Classification scores of shape (batch_size, 1)
        """
        # Extract features through convolutional layers
        h = self.net(x)  # (B, in_ch, H, W) -> (B, base*4, 1, 1)

        # Flatten spatial dimensions for linear layer
        h = h.flatten(1)  # (B, base*4, 1, 1) -> (B, base*4)

        # Generate final classification score
        return self.head(h)  # (B, base*4) -> (B, 1)
