"""
VoVNet FPS benchmark on a single Wormhole N150 (torch_xla execute path).

This is the simplest guaranteed way to get an FPS number on N150: runs the
model through torch_xla with the correct mark_step() timing pattern. Single
chip, no SPMD/data-parallel (that was only needed on N300's 2 chips).

NOTE: this measures the torch_xla EXECUTE path (default lowering). It is the
honest "what does the device do out of the box" number. The optimized
codegen + metal-trace path (reviewer's ~339 FPS) is a separate artifact —
see codegen_py_vovnet.py / the C++ standalone.

Run:
    python bench_vovnet_n150.py
"""
import os
os.environ["TT_VISIBLE_DEVICES"] = "0"   # N150 = single chip, device 0
os.environ["PJRT_DEVICE"] = "TT"

import time
import timm
import torch
import torch_xla
import torch_xla.core.xla_model as xm

model = timm.create_model("ese_vovnet19b_dw.ra_in1k", pretrained=True).eval().to(torch.bfloat16)
device = torch_xla.device()
model = model.to(device)

for batch in [1, 8, 16, 32]:
    x = torch.randn(batch, 3, 224, 224, dtype=torch.bfloat16).to(device)

    # Warmup (triggers compile). mark_step() forces the lazy IR to dispatch.
    for _ in range(5):
        with torch.no_grad():
            out = model(x)
        xm.mark_step()
    xm.wait_device_ops()

    # Benchmark.
    N = 30
    t0 = time.time()
    for _ in range(N):
        with torch.no_grad():
            out = model(x)
        xm.mark_step()
    xm.wait_device_ops()
    elapsed = time.time() - t0

    sanity = out.sum().cpu().item()          # forces last output to materialize
    fps = (N * batch) / elapsed
    print(f"batch={batch}: {(elapsed / N) * 1000:.2f} ms/iter, "
          f"{fps:.1f} FPS  (sanity={sanity:.2f})")
