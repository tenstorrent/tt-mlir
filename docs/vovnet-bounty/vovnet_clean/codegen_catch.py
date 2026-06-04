import os
os.environ["TT_VISIBLE_DEVICES"] = "0"
os.environ["PJRT_DEVICE"] = "TT"

import glob
import timm
import torch
import torch_xla
import torch_xla.core.xla_model as xm

torch_xla.set_custom_compile_options({
    "backend": "codegen_cpp",
    "export_path": "vovnet_cpp_catch",
    "export_tensors": True,
    "codegen_split_files": True,
    "optimization_level": 2,
})

model = timm.create_model("ese_vovnet19b_dw.ra_in1k", pretrained=True).eval().to(torch.bfloat16)
model.compile(backend="tt", options={"tt_legacy_compile": True})

device = xm.xla_device()
model = model.to(device)

x = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16).to(device)

try:
    print("running model with codegen flag...")
    output = model(x)
    xm.wait_device_ops()
    print(f"output shape: {output.shape}")
except Exception as e:
    print(f"\nException caught: {type(e).__name__}: {e}")
    print("Checking if codegen files were emitted anyway...")

files = glob.glob("vovnet_cpp_catch/**/*", recursive=True)
print(f"\nFiles in vovnet_cpp_catch/: {len(files)}")
for f in files[:20]:
    print(f"  {f}")
