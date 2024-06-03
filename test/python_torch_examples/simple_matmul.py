import torch
import ttmlir

class MM(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


m = MM()
compiled_m = torch.compile(m, backend=ttmlir.compile)
a = torch.randn(1, 128, 64)
b = torch.randn(1, 64, 128)
compiled_m(a, b)
