import os
os.environ["TT_VISIBLE_DEVICES"] = "0"
os.environ["PJRT_DEVICE"] = "TT"

import timm
import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Set codegen options WITHOUT calling model.compile()
torch_xla.set_custom_compile_options({
    "backend": "codegen_cpp",
    "export_path": "vovnet_cpp_bypass",
    "export_tensors": True,
    "codegen_split_files": True,
    "optimization_level": 2,
})

model = timm.create_model("ese_vovnet19b_dw.ra_in1k", pretrained=True).eval().to(torch.bfloat16)
device = xm.xla_device()
model = model.to(device)

x = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16).to(device)

print("running model() WITHOUT torch.compile...")
output = model(x)
xm.wait_device_ops()

print(f"output shape: {output.shape}")
print("DONE")
