// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="mesh-shape=1,2 enable-cpu-hoisted-const-eval=true" -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Multi-chip const-eval CPU hoisting. The const-eval subgraph is segmented
// around the all_gather barrier into two CPU-hoisted segments. Each segment is
// emitted as a torch-typed `_impl` body plus a ttnn-typed wrapper that defers
// to utils.execute_cpu_hoisted_function, which runs the body shard-by-shard
// over the mesh. The all_gather stays on device between the two calls.

// CHECK: # File: "consteval"

// Two segment bodies (add/multiply before the CCL, subtract/multiply after).
// CHECK-DAG: def cpu_hoisted_const_eval_{{.*}}_impl(
// CHECK-DAG: ttir_cpu.add(
// CHECK-DAG: ttir_cpu.subtract(

// Each hoisted segment defers to the shard-by-shard helper, passing its body
// and the device mesh shape.
// CHECK-DAG: utils.execute_cpu_hoisted_function({{.*}}, cpu_hoisted_const_eval_{{.*}}_impl, mesh_shape=(1, 2))
// CHECK-DAG: utils.execute_cpu_hoisted_function({{.*}}, cpu_hoisted_const_eval_{{.*}}_impl, mesh_shape=(1, 2))

// The device-side const-eval function calls the first segment, performs the
// all_gather on device, brings the result to host, then calls the second
// segment.
// CHECK-LABEL: def forward_const_eval_0(
// CHECK: cpu_hoisted_const_eval_{{.*}}(
// CHECK: ttnn.all_gather
// CHECK: ttnn.from_device
// CHECK: cpu_hoisted_const_eval_{{.*}}(
// CHECK: return
func.func @forward(
    %arg0: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg1: tensor<32x32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}
) -> tensor<64x32xf32> {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %mul = "ttir.multiply"(%add, %arg0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %g = "ttir.all_gather"(%mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xf32>) -> tensor<64x32xf32>
  %sub = "ttir.subtract"(%g, %g) : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  %mul2 = "ttir.multiply"(%sub, %g) : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  return %mul2 : tensor<64x32xf32>
}
