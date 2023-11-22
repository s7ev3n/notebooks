import torch
imort torch.nn as nn

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
    
    attn = torch.nn.functional.softmax(score, dim=-1)
    if dropout is not None:
        # TODO: Why dropout here, does the original paper has it ?
        attn = dropout(attn)

    return torch.matmul(attn, value), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1)：
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
        
        attn_value, attn_weight = attention(q, k, v, attn_mask=mask, dropout=self.dropout)
        # attn_value -> (b, num_heads, t, d_k), attn_weight -> (b, num_heads, t, t)
        attn_value = attn_value.transpose(-2, -3) # -> (b, t, num_heads, d_k)
        attn_value = attn_value.view(b, t, -1) # -> (b, t, num_heads * d_k)
        
        res = self.l(attn_value) # -> (b, t, d_model)
    
        return res 


class SublayerResidual(nn.Module):
    def __init__(self, sublayer, d_model=512, dropout=0.1):
        super(SublayerResidual, self).__init__()
        self.sublayer = sublayer
        self.ln = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.sublayer(x))

class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model=512,  d_f=2048):
        super(Pointwise, self).__init__()
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


