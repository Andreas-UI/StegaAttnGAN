import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseLayer(nn.Module):
    """
    Differentiable noise layer for robust steganographic training.

    Applies various types of noise and distortions to images during training
    to simulate real-world conditions and improve robustness of steganographic
    methods. Each noise type is applied probabilistically, making the training
    more diverse and the model more resilient to attacks.

    Noise types included:
    1. Gaussian Noise: Additive white noise
    2. Blur: Simple averaging filter (box blur)
    3. Dropout: Random spatial masking
    4. Resize: Downscaling followed by upscaling
    5. JPEG-like: Quantization to simulate compression artifacts

    Args:
        jpeg_prob (float, optional): Probability of applying JPEG-like quantization. Defaults to 0.3.
        gauss_prob (float, optional): Probability of applying Gaussian noise. Defaults to 0.5.
        blur_prob (float, optional): Probability of applying blur. Defaults to 0.3.
        drop_prob (float, optional): Probability of applying spatial dropout. Defaults to 0.3.
        resize_prob (float, optional): Probability of applying resize distortion. Defaults to 0.3.

    Input:
        x (torch.Tensor): Input images of shape (batch_size, channels, height, width)
                         Expected to be in range [-1, 1]

    Output:
        torch.Tensor: Noisy images of same shape as input, clamped to [-1, 1]

    Example:
        >>> noise_layer = NoiseLayer(gauss_prob=0.7, blur_prob=0.4)
        >>> x = torch.randn(4, 3, 32, 32) * 0.5  # Images in [-1, 1] range
        >>> noisy_x = noise_layer(x)
        >>> print(noisy_x.shape)  # torch.Size([4, 3, 32, 32])

    Note:
        - All operations are differentiable for end-to-end training
        - Noise parameters are randomly sampled within predefined ranges
        - Multiple noise types can be applied to the same image
        - Input images should be normalized to [-1, 1] range
    """

    def __init__(
        self,
        jpeg_prob=0.3,
        gauss_prob=0.5,
        blur_prob=0.3,
        drop_prob=0.3,
        resize_prob=0.3,
    ):
        super().__init__()

        # Store probabilities for each noise type
        self.jpeg_prob = jpeg_prob  # JPEG-like quantization probability
        self.gauss_prob = gauss_prob  # Gaussian noise probability
        self.blur_prob = blur_prob  # Blur probability
        self.drop_prob = drop_prob  # Spatial dropout probability
        self.resize_prob = resize_prob  # Resize distortion probability

    def forward(self, x):
        """
        Apply random noise and distortions to input images.

        Args:
            x (torch.Tensor): Input images in range [-1, 1] of shape (B, C, H, W)

        Returns:
            torch.Tensor: Distorted images in range [-1, 1] of shape (B, C, H, W)
        """
        # Gaussian Noise: Add random noise sampled from normal distribution
        if torch.rand(1).item() < self.gauss_prob:
            # Random standard deviation between 0.0 and 0.05
            std = torch.empty(1).uniform_(0.0, 0.05).item()
            x = x + std * torch.randn_like(x)

        # Blur: Apply simple box filter for smoothing
        if torch.rand(1).item() < self.blur_prob:
            k = 3  # Kernel size (3x3 box filter)
            # Create averaging kernel: all weights = 1/(k*k)
            weight = torch.ones(1, 1, k, k, device=x.device) / (k * k)
            # Repeat kernel for each input channel (depthwise convolution)
            weight = weight.repeat(x.size(1), 1, 1, 1)  # (C, 1, 3, 3)
            # Apply grouped convolution (separate kernel per channel)
            x = F.conv2d(x, weight, padding=k // 2, groups=x.size(1))

        # Spatial Dropout: Randomly mask out spatial regions
        if torch.rand(1).item() < self.drop_prob:
            # Create random mask: 90% pixels kept, 10% dropped
            mask = (torch.rand_like(x[:, :1]) > 0.1).float()  # (B, 1, H, W)
            x = x * mask  # Broadcasting applies mask to all channels

        # Resize Distortion: Downscale then upscale to simulate quality loss
        if torch.rand(1).item() < self.resize_prob:
            B, C, H, W = x.shape
            # Random scale factor between 0.5 and 1.0
            scale = float(torch.empty(1).uniform_(0.5, 1.0).item())
            # Calculate new dimensions (minimum 8x8 to avoid too small images)
            nh, nw = max(8, int(H * scale)), max(8, int(W * scale))
            # Downscale then upscale back to original size
            x = F.interpolate(
                F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        # JPEG-like Quantization: Simulate compression artifacts
        if torch.rand(1).item() < self.jpeg_prob:
            # Random quantization factor (higher = more compression)
            q = float(torch.empty(1).uniform_(32, 128).item())
            # Quantize: scale up, round, then scale back down
            x = torch.round(x * q) / q

        # Ensure output remains in valid range [-1, 1]
        return x.clamp(-1, 1)
