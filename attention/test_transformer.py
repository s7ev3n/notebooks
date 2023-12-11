import torch
from transformer import create_transformer, causal_masking

def test_create_transformer():
    src_vocab_size = 16
    tgt_vocab_size = 16

    model = create_transformer(src_vocab_size, tgt_vocab_size)
    x = torch.tensor([[1, 2, 3, 4 ]])
    y = torch.tensor([[6, 7, 8, 9 ]])
    seq_len = x.size(1)
    src_mask = torch.ones((seq_len, seq_len))
    tgt_mask = causal_masking(seq_len)

    out = model(x, src_mask, y, tgt_mask)

    print(out)


if __name__ == "__main__":
    test_create_transformer()
