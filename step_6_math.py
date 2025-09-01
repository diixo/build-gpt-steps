
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

B,T,C=4,8,2

x = torch.randn(B,T,C)
print(x.shape)


##############################################################

xbow = torch.zeros((B,T,C))

for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)

print(x[0])

print(xbow[0])

# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)

##############################################################

# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)

# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)


###############################################################
print(80 * "*")

# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))

wei = torch.zeros((T, T))

print(tril)
print(wei)

wei = wei.masked_fill(tril == 0, float('-inf'))
print(wei)

wei = F.softmax(wei, dim=-1)
print(wei)

xbow3 = wei @ x
torch.allclose(xbow, xbow3)

