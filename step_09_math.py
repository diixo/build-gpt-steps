import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

B, T, C=4,8,32

x = torch.randn(B, T, C)
print(x.shape)


# version 4: single Head perform of self-attention
head_size = 16

key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)      # B, T, head_size=16
q = query(x)    # B, T, head_size=16

wei = q @ k.transpose(-2, -1) * head_size**-0.5  # (B, T, 16) @ (B, 16, T) -->> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T, T))

# print(tril)
# print(wei)

wei = wei.masked_fill(tril == 0, float('-inf'))
# print(wei)

wei = F.softmax(wei, dim=-1)
v = value(x)
out = wei @ v

print("v:", v.shape)
print("out:", out.shape)
# print(wei)
# torch.allclose(xbow, xbow3)
