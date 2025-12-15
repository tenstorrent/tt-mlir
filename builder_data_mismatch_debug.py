#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
from builder.base.builder_apis import (
    load_mlir_file,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import execute_fb

# Configuration
mlir_file_path = "ttir_resnet/resnet_bf16_ttir.mlir"
target = "ttir"
system_desc_path = "ttrt-artifacts/system_desc.ttsys"

# Step 1: Load MLIR file
print(f"Loading {mlir_file_path}...")
with open(mlir_file_path, "r") as f:
    mlir_ir_string = f.read()

mlir_module, builder = load_mlir_file(mlir_ir_string, target=target)
print("MLIR module loaded successfully")

# Step 2: Compile to flatbuffer
print("\nCompiling to flatbuffer...")
pipeline_options = []
# pipeline_options.append("enable-bfp8-conversion=true")
mlir_path, goldens = compile_ttir_module_to_flatbuffer(
    mlir_module,
    builder,
    system_desc_path=system_desc_path,
    test_base="debug_test",
    output_root=".",
    target="ttnn",
    module_dump=True,
)
print(f"Compiled MLIR: {mlir_path}")

fb_path = f"{mlir_path}.ttnn"
print(f"Flatbuffer: {fb_path}")

# Step 3: Execute on device with golden verification
print("\nExecuting on device...")
golden_tensors = {}
for loc, golden in goldens.items():
    golden_tensors[loc] = builder._generate_golden_device_tensor(loc, golden)

golden_report = execute_fb(
    fb_path=fb_path,
    pcc=0.99,
    atol=1e-08,
    rtol=1e-05,
    disable_golden=False,
    device=None,
    check_atol=True,
    check_rtol=True,
    goldens=goldens,
    bypass_ops=builder._bypass_ops,
    enable_intermediate_verification=True,
)

# Step 4: Build and save report
print("\nBuilding report...")
report = {}
for loc, device_results in golden_report.items():
    operand = builder._loc_to_operand.get(loc)
    op_name = ""
    if operand is not None and hasattr(operand, "OPERATION_NAME"):
        op_name = getattr(operand, "OPERATION_NAME", "") or ""

    report[loc] = {
        "op_name": op_name,
        **device_results[0],
    }

# Save report
report_path = "golden_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to: {report_path}")
print(f"Total ops checked: {len(report)}")

# Print summary
passed = sum(1 for r in report.values() if r.get("result") == "pass")
failed = len(report) - passed
print(f"Passed: {passed}, Failed: {failed}")
