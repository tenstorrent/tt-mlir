"""
Python codegen for VoVNet (reviewer's recommended path).

Mirrors codegen_cpp_vovnet.py but emits a standalone PYTHON module instead of
C++. Easier tooling, and with export_tensors=True the real trained weights are
serialized to disk so you can validate accuracy vs torch-CPU (the C++ ones()
path cannot).

This GENERATES code + MLIR + weights — it does NOT print FPS. Output lands in
vovnet_py/:
    vovnet_py/irs/*.mlir     # vhlo -> shlo -> ttir -> ttnn (opt2 applied)
    vovnet_py/*.py           # generated standalone inference module
    vovnet_py/tensors/...    # serialized weights

Run:
    python codegen_py_vovnet.py

Then run the generated module to execute on-device. For an FPS number, wrap its
forward in a warmup + metal-trace + timed loop (this is how the reviewer hit
~339 FPS on N150), or use bench_vovnet_n150.py for the quick execute-path number.
"""
import os
os.environ["TT_VISIBLE_DEVICES"] = "0"   # N150 = single chip
os.environ["PJRT_DEVICE"] = "TT"

import timm
import torch
from tt_torch import codegen_py

model = timm.create_model("ese_vovnet19b_dw.ra_in1k", pretrained=True).eval().to(torch.bfloat16)
x = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)

print("running codegen_py...")
codegen_py(
    model,
    x,
    export_path="vovnet_py",
    compiler_options={
        "codegen_split_files": True,
        "optimization_level": 2,
        # Serialize real weights to disk for accuracy checks. If codegen_py
        # rejects this key in compiler_options, pass export_tensors=True as a
        # top-level kwarg instead (as in codegen_catch.py's custom options).
        "export_tensors": True,
    },
)
print("DONE -> vovnet_py/  (generated .py + tensors/ + irs/*.mlir)")
