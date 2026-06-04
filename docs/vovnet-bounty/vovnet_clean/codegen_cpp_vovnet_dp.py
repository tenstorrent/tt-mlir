import os
os.environ["TT_VISIBLE_DEVICES"] = "0"
os.environ["PJRT_DEVICE"] = "TT"
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

import timm
import torch
import numpy as np
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from tt_torch import codegen_cpp

xr.use_spmd()

num_devices = xr.global_runtime_device_count()
print("num_devices =", num_devices)

device = torch_xla.device()

model = timm.create_model("ese_vovnet19b_dw.ra_in1k", pretrained=True).eval().to(torch.bfloat16)
model = model.to(device)                          # model na XLA device (replikovan)

batch_size = num_devices * 4
x = torch.randn(batch_size, 3, 224, 224, dtype=torch.bfloat16).to(device)   # ulaz na device

device_ids = np.arange(num_devices)
mesh = Mesh(device_ids=device_ids, mesh_shape=(num_devices,), axis_names=("data",))
xs.mark_sharding(x, mesh, ("data", None, None, None))   # sad x JESTE XLA tensor

print("running codegen_cpp (data parallel)...")
codegen_cpp(
    model,
    x,
    export_path="vovnet_cpp_dp",
    compiler_options={
        "codegen_split_files": True,
        "optimization_level": 2,
    },
)
print("DONE")
