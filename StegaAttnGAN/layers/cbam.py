import torch.nn as nn
from StegaAttnGAN.layers import SEBlock, SpatialAttn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) for dual attention mechanism.

    CBAM combines channel attention and spatial attention in sequence to adaptively
    refine feature representations. It first applies channel attention to determine
    'what' to focus on, then spatial attention to determine 'where' to focus.

    The module sequentially applies:
    1. Channel Attention (SEBlock): Recalibrates channel-wise feature responses
    2. Spatial Attention: Recalibrates spatial feature responses

    Paper: "CBAM: Convolutional Block Attention Module" (https://arxiv.org/abs/1807.06521)

    Args:
        ch (int): Number of input/output channels

    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, ch, height, width)

    Output:
        torch.Tensor: Attention-refined features of shape (batch_size, ch, height, width)
        Features are recalibrated by both channel and spatial attention mechanisms.

    Example:
        >>> cbam = CBAM(ch=256)
        >>> x = torch.randn(4, 256, 32, 32)
        >>> output = cbam(x)
        >>> print(output.shape)  # torch.Size([4, 256, 32, 32])

    Note:
        The order matters: channel attention is applied first, followed by spatial attention.
        This design allows the model to first select important channels, then focus on
        important spatial locations within those channels.
    """

    def __init__(self, ch):
        super().__init__()

        # Channel Attention Module (SE Block)
        # Learns channel-wise attention weights using global average pooling
        # and two fully connected layers
        self.c = SEBlock(ch)

        # Spatial Attention Module
        # Learns spatial attention weights by aggregating channel information
        # using avg/max pooling and a 7x7 convolution
        self.s = SpatialAttn(ch)

    def forward(self, x):
        """
        Forward pass through CBAM with sequential attention application.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ch, height, width)

        Returns:
            torch.Tensor: Dual-attention refined features of shape (batch_size, ch, height, width)
        """
        # Step 1: Apply channel attention
        # Determines which channels are important
        x_channel_attended = self.c(x)

        # Step 2: Apply spatial attention to channel-attended features
        # Determines which spatial locations are important
        return self.s(x_channel_attended)
