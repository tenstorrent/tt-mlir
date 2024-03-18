import torch
import ttmlir

class Eltwise(torch.nn.Module):
    def forward(self, a, b):
        return a * b


m = Eltwise()
compiled_m = torch.compile(m, backend=ttmlir.compile)
a = torch.randn(64, 128)
b = torch.randn(64, 128)
compiled_m(a, b)
