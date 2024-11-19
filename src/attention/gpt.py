"""GPT model implementation based on nanoGPT."""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer import TransformerEncoderBlock, WordEmbedding, PositionEncoding, causal_masking

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_embed: int = 768
    n_head: int = 12
    dropout: float = 0.1

class GPT(nn.Module):
    def __init__(self, cfg):
        super(GPT, self).__init__()
        self.n_embed = cfg.n_embed
        self.n_layer = cfg.n_layer
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.vocab_size = cfg.vocab_size
        self.block_size = cfg.block_size # sequence length
        self.wte = WordEmbedding(vocab_size=self.vocab_size, 
                                 d_model=self.n_embed)
        self.pte = PositionEncoding(d_model=self.n_embed, 
                                    dropout=self.dropout,
                                    max_len=self.block_size)
        
        self.encoder_block = TransformerEncoderBlock(
            n_embed=self.n_embed,
            n_head=self.n_head,
            dropout=self.dropout)
        self.transformer = nn.ModuleDict(dict(
            wte = self.wte,
            pte = self.pte,
            encoder = nn.ModuleList([self.encoder_block for _ in range(self.n_layer)]),
            ln = nn.LayerNorm(cfg.n_embed)
        ))
        # language modeling head
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size, bias=False)
        # attn mask
        attn_mask = causal_masking(self.block_size)
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x, y=None):
        # x is tokenized index, (b, t)
        # y is label, (b, t)
        x = self.transformer.pte(self.transformer.wte(x))
        for block in self.transformer.encoder:
            x = block(x, self.attn_mask)
        x = self.transformer.ln(x)
        logits = self.lm_head(x) # (b, t, n_embed)

        if y is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), 
                                   y.view(-1))
        else:
            # inference-time optimization: only forward the lm_head on the 
            # very last embedding, (b, 1, c)
            logits = logits[:, [-1], :] # note: using list [-1] to preserve the time dim 
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_idx, max_new_tokens, temperature=1.0):
        "Take a input sequence of indices and complete the sequence."
        for _ in range(max_new_tokens):
            idx_cond = input_idx if input_idx.size(1) <= self.block_size else input_idx[:, :self.block_size]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # (b, c)
            prob = F.softmax(logits, dim=-1)
            # idx_next = F.argmax(prob, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1) # (b, 1)
            input_idx = torch.cat((idx_cond, idx_next), dim = 1)

        return input_idx
