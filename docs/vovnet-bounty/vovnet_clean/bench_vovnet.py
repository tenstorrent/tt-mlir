import os
os.environ["TT_VISIBLE_DEVICES"] = "0"           # 1 N300 board = 2 chips
os.environ["PJRT_DEVICE"] = "TT"
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"       # required for SPMD path

import timm
import torch
import time
import numpy as np
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

xr.set_device_type("TT")
xr.use_spmd()

num_devices = xr.global_runtime_device_count()
print(f"num_devices = {num_devices}")          # treba da bude 2

model = timm.create_model("ese_vovnet19b_dw.ra_in1k", pretrained=True).eval().to(torch.bfloat16)
device = torch_xla.device()
model = model.to(device)

# 1D data-parallel mesh
device_ids = np.arange(num_devices)
mesh = Mesh(device_ids=device_ids, mesh_shape=(num_devices,), axis_names=("data",))

# batch mora biti deljiv sa num_devices (=2)
for batch in [8, 16, 32]:
    x = torch.randn(batch, 3, 224, 224, dtype=torch.bfloat16).to(device)
    xs.mark_sharding(x, mesh, ("data", None, None, None))
    xm.mark_step()
    xm.wait_device_ops()

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            out = model(x)
        xm.mark_step()
    xm.wait_device_ops()

    # Benchmark
    N = 30
    t0 = time.time()
    for _ in range(N):
        with torch.no_grad():
            out = model(x)
        xm.mark_step()
    xm.wait_device_ops()
    elapsed = time.time() - t0

    sanity = out.sum().cpu().item()
    fps = (N * batch) / elapsed
    print(f"batch={batch}: {(elapsed/N)*1000:.2f}ms/iter, {fps:.1f} FPS  (sanity={sanity:.2f})")
