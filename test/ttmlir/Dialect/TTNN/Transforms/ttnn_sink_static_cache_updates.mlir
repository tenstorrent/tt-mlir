// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttnn-sink-static-cache-updates %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#cl_dram = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #ttnn.buffer_type<dram>>, <interleaved>>
#cl_l1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #ttnn.buffer_type<l1>>, <interleaved>>
#bf16_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>

// Positive case: CL cluster (delta→L1, add, dealloc, 2× DRAM copies) sits at the
// start of the function before model ops; the pass should sink it to just before
// the return so model ops come first in the output.
module {
  // CHECK-LABEL: func.func @sink_full_cluster
  func.func @sink_full_cluster(
      %arg0: tensor<1xsi32, #cl_dram>,
      %arg1: tensor<1xsi32, #cl_dram>,
      %arg2: tensor<32x32xbf16, #bf16_dram>,
      %arg3: tensor<32x32xbf16, #bf16_dram>
  ) -> (tensor<1xsi32, #cl_dram>, tensor<1xsi32, #cl_dram>, tensor<32x32xbf16, #bf16_dram>)
  attributes {tt.function_type = "forward_device"} {
    // CL cluster — should be sunk to just before the return.
    %delta_l1 = "ttnn.to_memory_config"(%arg1) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> : (tensor<1xsi32, #cl_dram>) -> tensor<1xsi32, #cl_l1>
    %cl_new = "ttnn.add"(%arg0, %delta_l1) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<1xsi32, #cl_dram>, tensor<1xsi32, #cl_l1>) -> tensor<1xsi32, #cl_l1>
    "ttnn.deallocate"(%delta_l1) <{force = false}> : (tensor<1xsi32, #cl_l1>) -> ()
    %dram1 = "ttnn.to_memory_config"(%cl_new) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1xsi32, #cl_l1>) -> tensor<1xsi32, #cl_dram>
    %dram2 = "ttnn.to_memory_config"(%cl_new) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1xsi32, #cl_l1>) -> tensor<1xsi32, #cl_dram>

    // Model op — must appear BEFORE the sunk cluster in the output.
    // CHECK: "ttnn.add"(%arg2, %arg3)
    %model_out = "ttnn.add"(%arg2, %arg3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #bf16_dram>, tensor<32x32xbf16, #bf16_dram>) -> tensor<32x32xbf16, #bf16_dram>

    // Sunk cluster then appears immediately before the return.
    // CHECK: "ttnn.to_memory_config"(%arg1)
    // CHECK: "ttnn.add"(%arg0,
    // CHECK: "ttnn.deallocate"
    // CHECK: "ttnn.to_memory_config"
    // CHECK: "ttnn.to_memory_config"
    // CHECK: return
    return %dram1, %dram2, %model_out : tensor<1xsi32, #cl_dram>, tensor<1xsi32, #cl_dram>, tensor<32x32xbf16, #bf16_dram>
  }
}

// Positive case: add directly uses block args (no delta→L1 step).
module {
  // CHECK-LABEL: func.func @sink_no_delta_l1
  func.func @sink_no_delta_l1(
      %arg0: tensor<1xsi32, #cl_dram>,
      %arg1: tensor<1xsi32, #cl_dram>,
      %arg2: tensor<32x32xbf16, #bf16_dram>,
      %arg3: tensor<32x32xbf16, #bf16_dram>
  ) -> (tensor<1xsi32, #cl_dram>, tensor<32x32xbf16, #bf16_dram>)
  attributes {tt.function_type = "forward_device"} {
    // add directly between two DRAM tensors — no deltaToL1 node.
    %cl_new = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<1xsi32, #cl_dram>, tensor<1xsi32, #cl_dram>) -> tensor<1xsi32, #cl_dram>
    %dram1 = "ttnn.to_memory_config"(%cl_new) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1xsi32, #cl_dram>) -> tensor<1xsi32, #cl_dram>

    // CHECK: "ttnn.add"(%arg2, %arg3)
    %model_out = "ttnn.add"(%arg2, %arg3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #bf16_dram>, tensor<32x32xbf16, #bf16_dram>) -> tensor<32x32xbf16, #bf16_dram>

    // CHECK: "ttnn.add"(%arg0, %arg1)
    // CHECK: "ttnn.to_memory_config"
    // CHECK: return
    return %dram1, %model_out : tensor<1xsi32, #cl_dram>, tensor<32x32xbf16, #bf16_dram>
  }
}

// Negative case: add result has a non-DRAM user (also consumed by a model op),
// so the cluster must NOT be moved.
module {
  // CHECK-LABEL: func.func @no_sink_add_has_non_dram_user
  func.func @no_sink_add_has_non_dram_user(
      %arg0: tensor<1xsi32, #cl_dram>,
      %arg1: tensor<1xsi32, #cl_dram>,
      %arg2: tensor<32x32xbf16, #bf16_dram>
  ) -> (tensor<1xsi32, #cl_dram>, tensor<32x32xbf16, #bf16_dram>)
  attributes {tt.function_type = "forward_device"} {
    %cl_new = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<1xsi32, #cl_dram>, tensor<1xsi32, #cl_dram>) -> tensor<1xsi32, #cl_dram>
    %dram1 = "ttnn.to_memory_config"(%cl_new) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1xsi32, #cl_dram>) -> tensor<1xsi32, #cl_dram>
    // cl_new also consumed by a non-DRAM op — cluster is not safe to sink.
    // CHECK: "ttnn.add"(%arg0, %arg1)
    %also_uses_cl = "ttnn.add"(%cl_new, %cl_new) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<1xsi32, #cl_dram>, tensor<1xsi32, #cl_dram>) -> tensor<1xsi32, #cl_dram>
    %model_out = "ttnn.add"(%arg2, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #bf16_dram>, tensor<32x32xbf16, #bf16_dram>) -> tensor<32x32xbf16, #bf16_dram>
    return %dram1, %model_out : tensor<1xsi32, #cl_dram>, tensor<32x32xbf16, #bf16_dram>
  }
}
