import torch
import _ttmlir_runtime as tt_runtime
from builder.base.builder_apis import load_mlir_file, split_mlir_file, compile_ttir_module_to_flatbuffer
from builder.base.builder_runtime import execute_fb

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 1)
device = tt_runtime.runtime.open_mesh_device(mesh_options)


# Load all 16 inputs in TTIR order (0.pt through 15.pt)
# These correspond to the arguments in mlp_mixer_sanity_ttir.mlir:
#  0: l__self___norm_bias (768xbf16)
#  1: l__self___norm_weight (768xbf16)
#  2: l__self___blocks_11_mlp_channels_fc2_bias (768xbf16)
#  3: l__self___blocks_11_mlp_channels_fc2_weight (768x3072xbf16)
#  4: l__self___blocks_11_mlp_channels_fc1_bias (3072xbf16)
#  5: l__self___blocks_11_mlp_channels_fc1_weight (3072x768xbf16)
#  6: l__self___blocks_11_norm2_bias (768xbf16)
#  7: l__self___blocks_11_norm2_weight (768xbf16)
#  8: l__self___blocks_11_mlp_tokens_fc2_bias (196xbf16)
#  9: l__self___blocks_11_mlp_tokens_fc2_weight (196x384xbf16)
# 10: l__self___blocks_11_mlp_tokens_fc1_bias (384xbf16)
# 11: l__self___blocks_11_mlp_tokens_fc1_weight (384x196xbf16)
# 12: l__self___blocks_11_norm1_bias (768xbf16)
# 13: l__self___blocks_11_norm1_weight (768xbf16)
# 14: args_1 (1x196x768xbf16) - input tensor
# 15: args_0 (1x196x768xbf16) - input tensor

inputs = [
    torch.load("mlp_sanity_inputs/0.pt"),   # norm_bias
    torch.load("mlp_sanity_inputs/1.pt"),   # norm_weight
    torch.load("mlp_sanity_inputs/2.pt"),   # blocks_11_mlp_channels_fc2_bias
    torch.load("mlp_sanity_inputs/3.pt"),   # blocks_11_mlp_channels_fc2_weight
    torch.load("mlp_sanity_inputs/4.pt"),   # blocks_11_mlp_channels_fc1_bias
    torch.load("mlp_sanity_inputs/5.pt"),   # blocks_11_mlp_channels_fc1_weight
    torch.load("mlp_sanity_inputs/6.pt"),   # blocks_11_norm2_bias
    torch.load("mlp_sanity_inputs/7.pt"),   # blocks_11_norm2_weight
    torch.load("mlp_sanity_inputs/8.pt"),   # blocks_11_mlp_tokens_fc2_bias
    torch.load("mlp_sanity_inputs/9.pt"),   # blocks_11_mlp_tokens_fc2_weight
    torch.load("mlp_sanity_inputs/10.pt"),  # blocks_11_mlp_tokens_fc1_bias
    torch.load("mlp_sanity_inputs/11.pt"),  # blocks_11_mlp_tokens_fc1_weight
    torch.load("mlp_sanity_inputs/12.pt"),  # blocks_11_norm1_bias
    torch.load("mlp_sanity_inputs/13.pt"),  # blocks_11_norm1_weight
    torch.load("mlp_sanity_inputs/14.pt"),  # args_1 (input tensor)
    torch.load("mlp_sanity_inputs/15.pt"),  # args_0 (input tensor)
]

print(f"Loaded {len(inputs)} input tensors:")
for idx, inp in enumerate(inputs):
    print(f"  Input {idx:2d}: shape={str(inp.shape):20s} dtype={inp.dtype}")

mlir_file_path = "irs/mlp_mixer_sanity_ttir.mlir"
with open(mlir_file_path, 'r') as f:
    mlir_ir_string = f.read()

mlir_module, builder = load_mlir_file(mlir_ir_string,golden_inputs = inputs, target="ttir")

builder_module_list = split_mlir_file(mlir_module, builder)

for idx, (split_module, split_builder) in enumerate(builder_module_list):
    print(split_module, flush=True)
    mlir_path, goldens = compile_ttir_module_to_flatbuffer(split_module, split_builder)
    fb_path = mlir_path + ".ttnn"

    try:
        golden_report = execute_fb(fb_path, goldens, device=device, enable_intermediate_verification=True)
        print(golden_report, flush=True)
    except Exception as e:
        print(f"Execution failed for module {idx} with error: {e}", flush=True)

tt_runtime.runtime.close_mesh_device(device)