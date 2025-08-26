import torch
from bspline_cuda import hello_cuda

N = 64
x = torch.zeros(N, device=torch.device('cuda'), dtype=torch.float32)

hello_cuda(x)
print(x)