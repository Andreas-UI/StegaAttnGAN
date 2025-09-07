import torch
import torch.nn as nn

from StegaAttnGAN.layers import ConvBlock, CBAM, ResidualBlock, SEBlock
from StegaAttnGAN.transformer import CrossAttn2D


class UNetGen(nn.Module):
    """
    U-Net Generator with Cross-Attention for Message-Conditioned Image Generation.

    A U-Net architecture enhanced with attention mechanisms and cross-modal fusion
    for steganographic image generation. The generator embeds text messages into
    images by conditioning the generation process on message token embeddings
    through cross-attention at multiple resolution levels.

    Architecture:
    - Encoder: Progressive downsampling with CBAM attention
    - Bottleneck: Residual processing with SE attention and cross-attention fusion
    - Decoder: Progressive upsampling with skip connections and cross-attention fusion

    Key Features:
    - Multi-scale message fusion via cross-attention at bottleneck and decoder levels
    - Skip connections preserve fine-grained details
    - Attention mechanisms (CBAM, SE) enhance feature representation
    - Residual connections in bottleneck for better gradient flow

    Args:
        in_ch (int, optional): Number of input image channels. Defaults to 3 (RGB).
        base (int, optional): Base number of feature channels. Defaults to 64.
        d_model (int, optional): Dimension for cross-attention (should match message encoder). Defaults to 256.
        nhead (int, optional): Number of attention heads in cross-attention. Defaults to 4.

    Input:
        img (torch.Tensor): Input cover images of shape (batch_size, in_ch, height, width)
        msg_tokens (torch.Tensor): Message embeddings of shape (batch_size, seq_len, d_model)

    Output:
        torch.Tensor: Generated stego images of shape (batch_size, in_ch, height, width)
                     Values are in range [-1, 1] due to tanh activation

    Example:
        >>> generator = UNetGen(in_ch=3, base=64, d_model=256, nhead=8)
        >>> img = torch.randn(4, 3, 32, 32)      # Cover images
        >>> msg = torch.randn(4, 20, 256)       # Message embeddings
        >>> stego = generator(img, msg)
        >>> print(stego.shape)  # torch.Size([4, 3, 32, 32])
        >>> print(stego.min(), stego.max())  # Values in [-1, 1]

    Note:
        - Cross-attention enables adaptive message embedding based on image content
        - Multi-scale fusion allows both coarse and fine-grained message integration
        - Skip connections preserve image quality while enabling message embedding
    """

    def __init__(self, in_ch=3, base=64, d_model=256, nhead=4):
        super().__init__()

        # ================== ENCODER ==================
        # First encoder block: Initial feature extraction with attention
        self.e1 = nn.Sequential(
            ConvBlock(
                in_ch, base
            ),  # Initial convolution: (B, 3, H, W) -> (B, base, H, W)
            ConvBlock(
                base, base
            ),  # Feature refinement: (B, base, H, W) -> (B, base, H, W)
            CBAM(base),  # Channel + spatial attention
        )

        # First downsampling: H×W -> H/2×W/2
        self.dn1 = nn.Conv2d(base, base * 2, 3, 2, 1)  # Stride=2 for downsampling

        # Second encoder block with attention
        self.e2 = nn.Sequential(
            ConvBlock(base * 2, base * 2),  # Feature processing: (B, 2*base, H/2, W/2)
            CBAM(base * 2),
        )  # Attention mechanism

        # Second downsampling: H/2×W/2 -> H/4×W/4
        self.dn2 = nn.Conv2d(base * 2, base * 4, 3, 2, 1)  # Stride=2 for downsampling

        # ================== BOTTLENECK ==================
        # Bottleneck processing with residual connections and attention
        self.bott = nn.Sequential(
            ConvBlock(base * 4, base * 4),  # Feature processing
            ResidualBlock(base * 4),  # Residual connection for gradient flow
            SEBlock(base * 4),  # Channel attention (squeeze-and-excitation)
        )

        # Cross-attention for message fusion in bottleneck
        self.ca_b = CrossAttn2D(base * 4, d_model, nhead)

        # ================== DECODER ==================
        # First decoder upsampling: H/4×W/4 -> H/2×W/2
        self.up2 = nn.ConvTranspose2d(
            base * 4, base * 2, 2, 2
        )  # Transpose conv for upsampling
        # First decoder block (processes concatenated features from skip connection)
        self.d2 = nn.Sequential(
            ConvBlock(
                base * 4, base * 2
            ),  # Process concatenated features: skip + upsampled
            CBAM(base * 2),
        )  # Attention mechanism

        # Cross-attention for message fusion at decoder level 2
        self.ca_2 = CrossAttn2D(base * 2, d_model, nhead)

        # Second decoder upsampling: H/2×W/2 -> H×W
        self.up1 = nn.ConvTranspose2d(
            base * 2, base, 2, 2
        )  # Transpose conv for upsampling

        # Second decoder block (processes concatenated features from skip connection)
        self.d1 = nn.Sequential(
            ConvBlock(
                base * 2, base
            ),  # Process concatenated features: skip + upsampled
            CBAM(base),
        )  # Attention mechanism

        # Cross-attention for message fusion at decoder level 1
        self.ca_1 = CrossAttn2D(base, d_model, nhead)

        # Output layer: Generate final stego image
        self.out = nn.Conv2d(
            base, in_ch, 1
        )  # 1×1 conv: (B, base, H, W) -> (B, in_ch, H, W)

    def forward(self, img: torch.Tensor, msg_tokens: torch.Tensor):
        """
        Generate steganographic images by embedding messages into cover images.

        Args:
            img (torch.Tensor): Cover images of shape (batch_size, in_ch, height, width)
            msg_tokens (torch.Tensor): Message embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Stego images of shape (batch_size, in_ch, height, width) in range [-1, 1]
        """
        # ================== ENCODER PASS ==================
        # Extract multi-scale features with skip connections
        x1 = self.e1(img)  # Level 1: (B, base, H, W) - full resolution
        x2 = self.e2(self.dn1(x1))  # Level 2: (B, base*2, H/2, W/2) - half resolution
        xb = self.bott(
            self.dn2(x2)
        )  # Bottleneck: (B, base*4, H/4, W/4) - quarter resolution

        # ================== MESSAGE FUSION IN BOTTLENECK ==================
        # Fuse message information at the deepest level
        xb = xb + self.ca_b(xb, msg_tokens)  # Residual connection with cross-attention

        # ================== DECODER PASS WITH SKIP CONNECTIONS ==================
        # Decoder Level 2: Upsample and fuse with skip connection
        y2 = self.up2(xb)  # Upsample: (B, base*4, H/4, W/4) -> (B, base*2, H/2, W/2)
        y2 = torch.cat([y2, x2], dim=1)  # Skip connection: (B, base*4, H/2, W/2)
        y2 = self.d2(y2)  # Process concatenated features: (B, base*2, H/2, W/2)
        y2 = y2 + self.ca_2(y2, msg_tokens)  # Message fusion with residual connection

        # Decoder Level 1: Upsample and fuse with skip connection
        y1 = self.up1(y2)  # Upsample: (B, base*2, H/2, W/2) -> (B, base, H, W)
        y1 = torch.cat([y1, x1], dim=1)  # Skip connection: (B, base*2, H, W)
        y1 = self.d1(y1)  # Process concatenated features: (B, base, H, W)
        y1 = y1 + self.ca_1(y1, msg_tokens)  # Message fusion with residual connection

        # ================== OUTPUT GENERATION ==================
        # Generate final steganographic image
        stego = torch.tanh(
            self.out(y1)
        )  # (B, base, H, W) -> (B, in_ch, H, W), range [-1, 1]
        
        return stego
