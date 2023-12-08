import torch
import torch.nn as nn

def attention(query, key, value, attn_mask=None, dropout=None):
    """Scaled Dot Product Attention.

    Attention(Q, K, V) = softmax(Q * K.T / sqrt(d_k)) * V
    
    Params:
        query: (b, t, d_k)
        key  : (b, t, d_k)
        value: (b, t, d_k)
    Returns:
        result: (b, t, d_k)
        attn  : (b, t, t)
    """
    
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(d_k) # (b, t, t)

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

    def foward(self, q, k, v, mask = None):
        # q, k, v -> (b, t, d_model)
        b, t, _ = q.size()

        q = self.W_q(q) # (b, t, d_model)
        k = self.W_k(k)
        v = self.W_v(v)
        
        q = torch.split(q, self.num_heads, dim = -1) # (b, t, num_heads, d_k)
        k = torch.split(k, self.num_heads, dim = -1)
        v = torch.split(v, self.num_heads, dim = -1)

        q = q.transpose(-2, -3) # (b, num_heads, t, d_k)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        
        x, attn = attention(q, k, v, attn_mask=mask, dropout=self.dropout)
        # x -> (b, num_heads, t, d_k), attn -> (b, num_heads, t, t)
        x = x.transpose(-2, -3) # -> (b, t, num_heads, d_k)
        x = x.view(b, t, -1) # -> (b, t, num_heads * d_k)
        
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
    def __init__(self, d_model=512,  d_f=2048):
        super(PointwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_f = d_f
        self.fc1 = nn.Linear(d_model, d_f, bias=True)
        self.fc2 = nn.Linear(d_f, d_model, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model, multi_head_attention, feedforward, dropout=0.1):
        self.attention = multi_head_attention 
        self.feedforward = feedforward
        self.residual1 = SublayerResidual(d_model, dropout)
        self.residual2 = SublayerResidual(d_model, dropout)

    def forward(self, x, mask):
        x = self.residual1(x, lambda x : self.attention(x, x, x, mask))
        x = self.residual2(x, self.feedforward)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feedforward, dropout=0.1):
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feedforward = feedforward
        self.residual1 = SublayerResidual(d_model, dropout)
        self.residual2 = SublayerResidual(d_model, dropout)
        self.residual3 = SublayerResidual(d_model, dropout)

    def forward(self, x, cross_x, mask_x, mask_cross):
        x = self.residual1(x, lambda x : self.self_attn(x, x, x, mask_x))
        x = self.residual2(x, lambda x : self.cross_attn(x, cross_x, cross_x, mask_cross))
        x = self.residual3(x, self.feedforward)

        return x

class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layer=6):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(n_layer)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, decoder_layer, n_layer=6):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(n_layer)])

    def forward(self, x, cross_x, mask_x, mask_cross):
        for layer in self.layers:
            x = layer(x, cross_x, mask_x, mask_cross)
        return x

class PositionEncoding(nn.Module):
    """PositionEncoding
    Positional encoding will sum with input embedding to give input embedding order.
    Positional encoding is given by the following equation:
    PE(pos, 2i)     = sin(pos / (10000 ^ (2i / d_model))) # for given position odd end even index are alternating
    PE(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d_model)))
    where pos is position in sequence and i and index in d_model.
    
    The positional encoding implementation is a matrix of (max_len, d_model), 
    this matrix is not updated by SGD, it is implemented as a buffer of nn.Module which 
    is the state of of the nn.Module.

    Detail 1:
    In addition, we apply dropout to the sums of the embeddings and the positional encodings 
    in both the encoder and decoder stacks. For the base model, we use a rate of P_drop = 0.1
    """
    def __init__(self, max_len, d_model, dropout=0.1):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        pos = torch.arange(0, self.max_len).unsqueeze(1) # (max_len, 1)
        demonitor = torch.pow(10000, torch.arange(0, self.max_len, 2) / self.d_model) # pos/demonitor is broadcastable
        
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

def masking(seq_len):
    """Masking of self-attention.
    The masking has many names: causal masking, look ahead masking, subsequent masking
    and decoder masking, etc. But the main purpose is the same, mask out after the position i 
    to prevent leaking of future information in the transformer decoder.
    Usually, the mask is a triangular matrix where the elements below diagnal is True and 
    above is False. 

    Args:
        seq_len (int): sequence length 
    """

    mask = torch.triu(torch.ones((1, seq_len,seq_len)), diagonal=1).type(torch.int8)
    
    return mask == 0


class Transformer(nn.Module):
    """Transformer.

    """
    
    def __init__(self, ):
        super(Transformer, self).__init__()

    