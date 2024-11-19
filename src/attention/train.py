import os
import numpy as np
import pickle
import torch
from torch.nn import functional as F
from gpt import GPT, GPTConfig

# data prep
dataset = 'shakespeare_char'
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
vocab_size=None
meta_p = os.path.join(data_dir, 'meta.pkl')
with open(meta_p, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

# model
model_args = dict(
    n_layer = 6,
    n_head = 8,
    n_embed = 512,
    block_size = 1024,
    dropout = 0.1,
    vocab_size = 50257 # GPT-2 vocab size
)
if vocab_size is not None:
    model_args["vocab_size"] = vocab_size
cfg = GPTConfig(**model_args)
model = GPT(cfg)
device = 'cuda'
model.to(device)
print(model)

# training config
max_iters = 10000
lr = 1e-4
wd = 1e-3
batch_size = 32

# train
def get_batch(batch_size, split):
    data = train_data if split == 'train' else val_data
    indices = torch.randint((len(data)-model_args['block_size']), (batch_size,))
    # Training a GPT is predicting the next token, so label is offset by 1
    x = torch.stack([torch.from_numpy((data[i : i + model_args['block_size']]).astype(np.int64)) for i in indices])
    y = torch.stack([torch.from_numpy((data[i+1: i+1+model_args['block_size']]).astype(np.int64)) for i in indices])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True) 
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# optim
optim = torch.optim.Adam(model.parameters(), lr, weight_decay=wd)
model.train()
for i in range(max_iters):
    x, y = get_batch(batch_size, 'train')
    _, loss = model(x, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # show training loss
    if i % 100 == 0:
        print(f'iter {i} | loss {loss.item()}')

    # save 
    if i & i % 1000 == 0:
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(model.state_dict(), f'checkpoints/{dataset}_{i}.pt')