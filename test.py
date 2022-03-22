import torch

a = torch.arange(1*2*3).view(1, 2, 3)
print(a)
b = torch.sum(a)
print(b)