// RUN: ttmlir-opt --const-eval-hoist-transform %s | FileCheck %s

// This test verifies that standalone creation ops with CanExecuteOnHostTrait
// (like ttnn.full, ttnn.arange, ttnn.constant) are hoisted into const-eval subgraphs
// even when consumed by non-const-eval ops, enabling caching across inference runs.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_1d = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // Test 1: Standalone ttnn.full op used by non-const-eval op
  // The full op should be hoisted into its own const-eval subgraph

  // CHECK-LABEL: func.func @standalone_full_const_eval_0
  // CHECK: %[[DEVICE:.*]] = "ttnn.get_device"()
  // CHECK: %[[FULL:.*]] = "ttnn.full"(%[[DEVICE]])
  // CHECK-SAME: fill_value = 0.000000e+00 : f32
  // CHECK: return %[[FULL]]

  // CHECK-LABEL: func.func @standalone_full(
  func.func @standalone_full(%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x32xbf16, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[CACHED:.*]] = ttcore.load_cached(@standalone_full_const_eval_0, [])
    %full = "ttnn.full"(%device) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 0.000000e+00 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    // CHECK: %[[RESULT:.*]] = "ttnn.add"(%arg0, %[[CACHED]])
    %result = "ttnn.add"(%arg0, %full) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    return %result : tensor<32x32xbf16, #ttnn_layout>
  }

  // Test 2: Multiple standalone creation ops
  // Each should get its own const-eval subgraph

  // CHECK-LABEL: func.func @multiple_creation_ops_const_eval_0
  // CHECK: %[[DEVICE0:.*]] = "ttnn.get_device"()
  // CHECK: "ttnn.full"(%[[DEVICE0]])
  // CHECK-SAME: fill_value = 1.000000e+00 : f32

  // CHECK-LABEL: func.func @multiple_creation_ops_const_eval_1
  // CHECK: %[[DEVICE1:.*]] = "ttnn.get_device"()
  // CHECK: "ttnn.full"(%[[DEVICE1]])
  // CHECK-SAME: fill_value = 2.000000e+00 : f32

  // CHECK-LABEL: func.func @multiple_creation_ops(
  func.func @multiple_creation_ops(%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x32xbf16, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[CACHED1:.*]] = ttcore.load_cached(@multiple_creation_ops_const_eval_0, [])
    %full1 = "ttnn.full"(%device) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 1.000000e+00 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    // CHECK: %[[CACHED2:.*]] = ttcore.load_cached(@multiple_creation_ops_const_eval_1, [])
    %full2 = "ttnn.full"(%device) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 2.000000e+00 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    // CHECK: %[[ADD1:.*]] = "ttnn.add"(%arg0, %[[CACHED1]])
    %add1 = "ttnn.add"(%arg0, %full1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    // CHECK: %[[ADD2:.*]] = "ttnn.add"(%[[ADD1]], %[[CACHED2]])
    %add2 = "ttnn.add"(%add1, %full2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    return %add2 : tensor<32x32xbf16, #ttnn_layout>
  }

  // Test 3: Creation op consumed by const-eval op
  // The creation op should be merged into the const-eval subgraph (existing behavior)

  // CHECK-LABEL: func.func @merged_with_const_eval_const_eval_0
  // CHECK: %[[DEVICE:.*]] = "ttnn.get_device"()
  // CHECK: %[[FULL_INNER:.*]] = "ttnn.full"(%[[DEVICE]])
  // CHECK-SAME: fill_value = 5.000000e+00 : f32
  // CHECK: %[[SUB:.*]] = "ttnn.subtract"(%arg0, %[[FULL_INNER]])
  // CHECK: return %[[SUB]]

  // CHECK-LABEL: func.func @merged_with_const_eval(
  func.func @merged_with_const_eval(
      %arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>},
      %arg1: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // The full op and subtract should be in the same const-eval subgraph
    // CHECK: %[[CACHED:.*]] = ttcore.load_cached(@merged_with_const_eval_const_eval_0, [%arg1])
    %full = "ttnn.full"(%device) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 5.000000e+00 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    %sub = "ttnn.subtract"(%arg1, %full) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    // CHECK: %[[RESULT:.*]] = "ttnn.add"(%arg0, %[[CACHED]])
    %result = "ttnn.add"(%arg0, %sub) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    return %result : tensor<32x32xbf16, #ttnn_layout>
  }

  // Test 4: ttnn.arange operation (no device operand)
  // Should be hoisted as a standalone const-eval subgraph

  // CHECK-LABEL: func.func @arange_op_const_eval_0
  // CHECK: %[[ARANGE:.*]] = "ttnn.arange"()
  // CHECK-SAME: end = 32
  // CHECK-SAME: start = 0
  // CHECK-SAME: step = 1
  // CHECK: return %[[ARANGE]]

  // CHECK-LABEL: func.func @arange_op(
  func.func @arange_op(%arg0: tensor<32xbf16, #ttnn_layout_1d> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32xbf16, #ttnn_layout_1d> {
    // CHECK: %[[CACHED:.*]] = ttcore.load_cached(@arange_op_const_eval_0, [])
    %arange = "ttnn.arange"() <{dtype = #ttcore.supportedDataTypes<bf16>, end = 32 : i64, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, start = 0 : i64, step = 1 : i64}> : () -> tensor<32xbf16, #ttnn_layout_1d>
    // CHECK: %[[RESULT:.*]] = "ttnn.add"(%arg0, %[[CACHED]])
    %result = "ttnn.add"(%arg0, %arange) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32xbf16, #ttnn_layout_1d>, tensor<32xbf16, #ttnn_layout_1d>) -> tensor<32xbf16, #ttnn_layout_1d>
    return %result : tensor<32xbf16, #ttnn_layout_1d>
  }

  // Test 5: ttnn.constant operation
  // Should be hoisted as a standalone const-eval subgraph

  // CHECK-LABEL: func.func @constant_op_const_eval_0
  // CHECK: %[[DEVICE:.*]] = "ttnn.get_device"()
  // CHECK: %[[CONST:.*]] = "ttnn.constant"(%[[DEVICE]])
  // CHECK-SAME: value = dense<3.000000e+00>
  // CHECK: return %[[CONST]]

  // CHECK-LABEL: func.func @constant_op(
  func.func @constant_op(%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x32xbf16, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[CACHED:.*]] = ttcore.load_cached(@constant_op_const_eval_0, [])
    %constant = "ttnn.constant"(%device) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, value = dense<3.000000e+00> : tensor<32x32xbf16>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    // CHECK: %[[RESULT:.*]] = "ttnn.multiply"(%arg0, %[[CACHED]])
    %result = "ttnn.multiply"(%arg0, %constant) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    return %result : tensor<32x32xbf16, #ttnn_layout>
  }

}
