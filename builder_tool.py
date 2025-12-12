import sys
import os

from builder.base.builder_apis import load_mlir_file
import _ttmlir_runtime as tt_runtime

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 1)
device = tt_runtime.runtime.open_mesh_device(mesh_options)

mlir_file_path = "builder_tool/irs/mlp_mixer_sanity_ttir.mlir"
with open(mlir_file_path, 'r') as f:
    mlir_ir_string = f.read()

mlir_module, builder = load_mlir_file(mlir_ir_string, target="ttir")

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