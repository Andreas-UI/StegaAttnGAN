import torch

PAD_TOKEN = 0  # reserve token ID 0 for padding


def rand_tokens_with_pad(B: int, real_len: int, max_len: int, vocab: int):
    """
    Returns (tokens, pad_mask)
    - tokens: shape [B, max_len], with padding tokens after real_len
    - pad_mask: shape [B, max_len], True where padding
    """
    tokens = torch.full((B, max_len), PAD_TOKEN, dtype=torch.long)
    if real_len > 0:
        tokens[:, :real_len] = torch.randint(
            1, vocab, (B, real_len)
        )  # avoid using PAD_TOKEN in data
    pad_mask = tokens.eq(PAD_TOKEN)
    return tokens, pad_mask

def to_minus1_1(t):
    return t * 2.0 - 1.0

def to_0_1(x):
    return (x + 1.0) / 2.0