import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A basic convolutional block consisting of Conv2d -> BatchNorm2d -> LeakyReLU.

    This is a common building block used in CNN architectures that combines
    convolution, normalization, and activation in a single module.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
        k (int, optional): Kernel size for convolution. Defaults to 3.
        s (int, optional): Stride for convolution. Defaults to 1.
        p (int, optional): Padding for convolution. Defaults to 1.
        norm (bool, optional): Whether to apply batch normalization. Defaults to True.

    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, in_ch, height, width)

    Output:
        torch.Tensor: Output tensor of shape (batch_size, out_ch, height_out, width_out)
        where height_out = floor((height + 2*p - k) / s) + 1
        and width_out = floor((width + 2*p - k) / s) + 1

    Example:
        >>> conv_block = ConvBlock(in_ch=64, out_ch=128)
        >>> x = torch.randn(4, 64, 32, 32)
        >>> output = conv_block(x)
        >>> print(output.shape)  # torch.Size([4, 128, 32, 32])
    """

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm=True):
        super().__init__()

        # 2D Convolution layer
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)

        # Batch normalization layer (or Identity if norm=False)
        self.norm = nn.BatchNorm2d(out_ch) if norm else nn.Identity()

        # LeakyReLU activation with negative slope of 0.1
        # inplace=True saves memory by modifying input tensor directly
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        """
        Forward pass through the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_ch, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_ch, height_out, width_out)
        """
        # Apply convolution -> normalization -> activation in sequence
        return self.act(self.norm(self.conv(x)))
