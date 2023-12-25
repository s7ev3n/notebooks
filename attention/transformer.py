"""
Transformer implementation for understanding and practice.
"""

import copy
import math
import torch
import torch.nn as nn

def attention(query, key, value, attn_mask=None, dropout=None):
    """Scaled Dot Product Attention.

    Attention(Q, K, V) = softmax(Q * K.T / sqrt(d_k)) * V

    Attetion operation: a query vector (1, c) calcuates its similarity with
    a sequence vectors (t, c), and obtained its output (1,c) by a weighted 
    sum over a sequence of value vectors (t, c), where "weighted" is the 
    similarity. If you have a sequence of query vectors (t, c), the output
    is a sequence (t, c), where each position (1, c) in the output is the 
    weighted sum given by the key and value sequence vectors.

    Example, let us see the attention weight and value vector multiply matrix,
    
    attention weight:
    [[1.0 , 0.0 , 0.0],
     [0.5 , 0.5 , 0.0],
     [0.33, 0.33, 0.33]]

    value vector (t, c), each row is value vector
    [[1, 2],
     [4, 5],
     [7, 8]]
    
    The result is 
    [[1.0, 2.0],
     [2.5, 3.5],
     [4.0, 5.0]]

    The element (column) of specific row in attention weight sepcifies how each row of
    value vector is summed to obtain specific row of output. For instance, in the second 
    row of attention weight is [0.5, 0.5, 0.0] it specifies the second sequence in the 
    output is obtained by 0.5 the first sequence and 0.5 the second sequence of in the 
    value vector.
    PS: see video 
    https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=2533s

    Params:
        query: (b, t, d_k)
        key  : (b, t, d_k)
        value: (b, t, d_k)
    Returns:
        result: (b, t, d_k)
        attn  : (b, t, t)
    """
    
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k) # (b, t, t)

    if attn_mask is not None:
        # NOTE: Why set mask position to -np.inf ?
        # 1. Make sure masking position has no effect, set to 0 DO NOT lead to probability 0 using softmax!
        # 2. Softmax will give close to 0.0 prob to -np.inf but not 0.0 to avoid gradient vanishing
        # 3. For computation stability, to avoid underflow
        score = score.masked_fill(attn_mask == 0, -1e9)
    
    attn = nn.functional.softmax(score, dim=-1)
    if dropout is not None:
        # TODO: Why dropout here, does the original paper has it ?
        attn = dropout(attn)

    return torch.matmul(attn, value), attn

def causal_masking(seq_len):
    """Masking of self-attention.

    The masking has many names: causal masking, look ahead masking, subsequent masking
    and decoder masking, etc. But the main purpose is the same, mask out after the 
    position i to prevent leaking of future information in the transformer decoder. 
    Usually, the mask is a triangular matrix where the elements below diagnal is True
    and above is False. 

    Args:
        seq_len (int): sequence length 
    """

    mask = torch.triu(torch.ones((1, seq_len,seq_len)), diagonal=1).type(torch.int8)
    
    return mask == 0

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.l = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask = None):
        # q, k, v -> (b, t, d_model)
        b, t, d_model = q.size()

        q = self.W_q(q) # (b, t, d_model)
        k = self.W_k(k)
        v = self.W_v(v)
        
        q = q.view(b, t, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        k = k.view(b, t, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        v = v.view(b, t, self.num_heads, d_model // self.num_heads).transpose(1, 2) # (b, num_heads, t, d_k)
        
        x, attn = attention(q, k, v, attn_mask=mask, dropout=self.dropout)
        # x -> (b, num_heads, t, d_k), attn -> (b, num_heads, t, t)
        x = x.transpose(1, 2) # -> (b, t, num_heads, d_k)
        # it is necessary to add contiguous here
        x = x.contiguous().view(b, t, d_model) # -> (b, t, num_heads * d_k)
        
        res = self.l(x) # -> (b, t, d_model)
    
        return res 


class SublayerResidual(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super(SublayerResidual, self).__init__()
        self.ln = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Detail 1:
        Note implementation here is pre-norm formulation:
            x + sublayer(LayerNorm(x))
        Origin paper is LayerNorm(x+sublayer(x)) which is called post-norm. 
        There are literatures about the pros and cons of pre-norm and post-norm[1,2].

        Detail 2:
        We apply dropout to the output of each sub-layer, before it is added to the 
        sub-layer input and normalized.

        Reference: 
        1. https://youtu.be/kCc8FmEb1nY?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=5723
        2. https://kexue.fm/archives/9009
        """
        return x + self.dropout(sublayer(self.ln(x)))

class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_f=2048, dropout=0.1):
        super(PointwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_f = d_f
        self.fc1 = nn.Linear(d_model, d_f, bias=True)
        self.fc2 = nn.Linear(d_f, d_model, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # Note: The Annotated Transformer add a dropout here before fc2
        # original paper seems not
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, multi_head_attention, feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = multi_head_attention 
        self.feedforward = feedforward
        self.residual1 = SublayerResidual(d_model, dropout)
        self.residual2 = SublayerResidual(d_model, dropout)
        self.size = d_model

    def forward(self, x, mask):
        x = self.residual1(x, lambda x : self.attention(x, x, x, mask))
        x = self.residual2(x, self.feedforward)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feedforward = feedforward
        self.residual1 = SublayerResidual(d_model, dropout)
        self.residual2 = SublayerResidual(d_model, dropout)
        self.residual3 = SublayerResidual(d_model, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.residual1(x, lambda x : self.self_attn(x, x, x, src_mask))
        x = self.residual2(x, lambda x : self.cross_attn(x, memory, memory, tgt_mask))
        x = self.residual3(x, self.feedforward)

        return x

class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layer=6):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(n_layer)])
        # Note: The Annotated Transformer add a layer norm here
        # original paper seems not explictly said this layer norm
        self.ln = nn.LayerNorm(encoder_layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.ln(x)

class Decoder(nn.Module):
    def __init__(self, decoder_layer, n_layer=6):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(n_layer)])

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

class PositionEncoding(nn.Module):
    """Position Encoding.

    Positional encoding will sum with input embedding to give input embedding order.
    Positional encoding is given by the following equation:
    
    PE(pos, 2i)     = sin(pos / (10000 ^ (2i / d_model))) # for given position odd end even index are alternating
    PE(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d_model)))
    where pos is position in sequence and i is index along d_model.
    
    The positional encoding implementation is a matrix of (max_len, d_model), 
    this matrix is not updated by SGD, it is implemented as a buffer of nn.Module which 
    is the state of of the nn.Module.

    Detail 1:
    In addition, we apply dropout to the sums of the embeddings and the positional encodings 
    in both the encoder and decoder stacks. For the base model, we use a rate of P_drop = 0.1
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        pos = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1)
        demonitor = torch.pow(10000, torch.arange(0, d_model, 2) / d_model) # pos/demonitor is broadcastable
        
        pe[:, 0::2] = torch.sin(pos / demonitor)
        pe[:, 1::2] = torch.cos(pos / demonitor)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (b, t, d_model)
        # self.pe[:, :x.size(1)] will return a new tensor, not buffer anymore
        # by default the new tensor's requires_grad is Fasle, but here we refer
        # to The Annotated Transformer, use in_place requires_grad_(False)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) # max_len is much longer than t
        return self.dropout(x)

class Embedding(nn.Module):
    """Embedding tokens.

    Transformer is first applied to language modelling. Language has finite words and symbols,
    therefore you could build a large vocabulary to hold them, every sentence is a combination
    of element in the vocabulary. Every word or symbol in the vocabulary is represented by a 
    learned vector which is called embedding, the embedding is learned through training. 

    Detail 1:
    In the embedding layers, we multiply those weights by sqrt(d_model)
    """
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding_table = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x : (b, t)
        out = self.embedding_table(x) * math.sqrt(self.d_model)
        return out

class Transformer(nn.Module):
    """Transformer.

    Args:
        encoder: Transformer encoder part consist of 6 encoder layer
        decoder: Transformer decoder part consist of 6 decoder layer
        src_embed_module: encoder embedding table + position encoding
        tgt_embed_module: decoder embedding table + position encoding, if the src and tgt are different languages
    """
    
    def __init__(self, encoder, decoder, 
                 src_embed_module, tgt_embed_module):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed_module = src_embed_module
        self.tgt_embed_module = tgt_embed_module

    def forward(self, x, src_mask, tgt, tgt_mask):
        x = self.encoder(self.src_embed_module(x), src_mask)
        x = self.decoder(self.tgt_embed_module(tgt), x, src_mask, tgt_mask)

        return x
    

def create_transformer(src_vocab_size, # source language vocabulary 
                       tgt_vocab_size, # target language vocabulary
                       d_model = 512, 
                       d_f = 2048, 
                       n_heads = 8,
                       n_layer = 6,
                       dropout = 0.1):

    multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=n_heads, dropout=dropout)
    feed_forward = PointwiseFeedForward(d_model=d_model, d_f=d_f)
    position_encoding = PositionEncoding(d_model=d_model, dropout=dropout)

    clone = copy.deepcopy

    encoder_layer = EncoderLayer(d_model=d_model, 
                                 multi_head_attention=clone(multi_head_attention), 
                                 feedforward=clone(feed_forward),
                                 dropout=dropout)
    encoder = Encoder(encoder_layer=encoder_layer, n_layer=n_layer)

    decoder_layer = DecoderLayer(d_model=d_model,
                                 self_attn=clone(multi_head_attention),
                                 cross_attn=clone(multi_head_attention),
                                 feedforward=clone(feed_forward),
                                 dropout=dropout)
    decoder = Decoder(decoder_layer=decoder_layer, n_layer=n_layer)

    transformer_model = Transformer(
        src_embed_module=nn.Sequential(Embedding(src_vocab_size, d_model), clone(position_encoding)),
        tgt_embed_module=nn.Sequential(Embedding(tgt_vocab_size, d_model), clone(position_encoding)),
        encoder=encoder,
        decoder=decoder
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    # From The Annotated Transfomer
    for p in transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer_model

