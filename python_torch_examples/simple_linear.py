import torch
import ttmlir

m = torch.nn.Linear(64, 128)
compiled_m = torch.compile(m, backend=ttmlir.compile)
act = torch.randn(32, 64)
compiled_m(act)
