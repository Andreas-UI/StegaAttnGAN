import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer models using sinusoidal functions.

    Adds positional information to input embeddings since Transformers lack
    inherent notion of sequence order. Uses sine and cosine functions of different
    frequencies to create unique positional encodings for each position.

    The encoding uses the formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    where pos is the position and i is the dimension index.

    Paper: "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)

    Args:
        d_model (int): The model dimension (embedding size)
        max_len (int, optional): Maximum sequence length to pre-compute. Defaults to 4096.

    Input:
        x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)

    Output:
        torch.Tensor: Input embeddings with added positional encoding of shape
                     (batch_size, seq_len, d_model)

    Example:
        >>> pe = PositionalEncoding(d_model=512, max_len=1000)
        >>> x = torch.randn(32, 100, 512)  # (batch_size, seq_len, d_model)
        >>> output = pe(x)
        >>> print(output.shape)  # torch.Size([32, 100, 512])

    Note:
        Positional encodings are pre-computed and stored as a buffer, making them
        parameter-free and not updated during training.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)

        # Create position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)

        # Compute division term for sinusoidal functions
        # div_term = 1 / (10000^(2i/d_model)) for i in [0, 1, 2, ..., d_model//2-1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model//2,)

        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model//2)

        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model//2)

        # Add batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)  # (1, L, D)

        # Register as buffer (not a parameter, won't be updated during training)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Embeddings with positional encoding of shape (batch_size, seq_len, d_model)
        """
        # Extract sequence length from input
        L = x.size(1)  # seq_len

        # Add positional encoding: x + PE[:, :L]
        # Broadcasting handles batch dimension automatically
        return x + self.pe[:, :L]  # (B, L, D) + (1, L, D) -> (B, L, D)
