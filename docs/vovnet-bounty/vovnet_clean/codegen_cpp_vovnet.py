import os
os.environ["TT_VISIBLE_DEVICES"] = "0"
os.environ["PJRT_DEVICE"] = "TT"

import timm
import torch
from tt_torch import codegen_cpp

model = timm.create_model("ese_vovnet19b_dw.ra_in1k", pretrained=True).eval().to(torch.bfloat16)
x = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)

print("running codegen_cpp...")
codegen_cpp(
    model,
    x,
    export_path="vovnet_cpp",
    compiler_options={
        "codegen_split_files": True,
        "optimization_level": 2,
    },
)
print("DONE")
