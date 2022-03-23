import torch

a = torch.ones((2, 2, 1))
b = torch.randn((1, 2, 2, 1))

print(a)
print(b)
print(a + b)
print(torch.add(b, a))

print(1/7)  
print(torch.div(1, 7))