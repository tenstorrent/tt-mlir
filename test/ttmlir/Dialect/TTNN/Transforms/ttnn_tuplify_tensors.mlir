// RUN: ttmlir-opt --ttnn-canonicalize-function-arguments --ttnn-reorder-function-arguments --ttnn-tuplify-tensors %s | FileCheck %s --check-prefix=CHECK
// RUN: ttmlir-opt --ttnn-tuplify-tensors="tuplify-mode=target-module" %s | FileCheck %s --check-prefix=TARGET-MODULE-CHECK

// CHECK-LABEL: func.func @test_input_parameter_split
// CHECK-SAME: (%[[INPUT_TUPLE:.*]]: tuple<tensor<64x128xf32>, tensor<32x64xf32>> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_positions = #ttcore.original_arg_positions<[0, 1]>}
// CHECK-SAME:  %[[PARAM_TUPLE:.*]]: tuple<tensor<128x256xf32>, tensor<256x512xf32>> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.original_arg_positions = #ttcore.original_arg_positions<[2, 3]>})
// CHECK-SAME: -> tuple<tensor<64x512xf32>, tensor<32x64xf32>>
func.func @test_input_parameter_split(
    %arg0: tensor<64x128xf32> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg1: tensor<32x64xf32> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg2: tensor<128x256xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg3: tensor<256x512xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>})
    -> (tensor<64x512xf32>, tensor<32x64xf32>) {
  // CHECK: %[[GET0:.*]] = ttcore.get_tuple_element %[[INPUT_TUPLE]][0]
  // CHECK: %[[GET1:.*]] = ttcore.get_tuple_element %[[INPUT_TUPLE]][1]
  // CHECK: %[[GET2:.*]] = ttcore.get_tuple_element %[[PARAM_TUPLE]][0]
  // CHECK: %[[GET3:.*]] = ttcore.get_tuple_element %[[PARAM_TUPLE]][1]
  // CHECK: %[[MATMUL_0:.*]] = "ttnn.matmul"(%[[GET0]], %[[GET2]])
  %0 = "ttnn.matmul"(%arg0, %arg2) : (tensor<64x128xf32>, tensor<128x256xf32>) -> tensor<64x256xf32>
  // CHECK: %[[MATMUL_1:.*]] = "ttnn.matmul"(%[[MATMUL_0]], %[[GET3]])
  %1 = "ttnn.matmul"(%0, %arg3) : (tensor<64x256xf32>, tensor<256x512xf32>) -> tensor<64x512xf32>
  // CHECK: %[[TUPLE_RESULT:.*]] = ttcore.tuple %[[MATMUL_1]], %[[GET1]]
  // CHECK: return %[[TUPLE_RESULT]] : tuple<tensor<64x512xf32>, tensor<32x64xf32>>
  return %1, %arg1 : tensor<64x512xf32>, tensor<32x64xf32>
}

// CHECK-LABEL: func.func @test_default_type
// CHECK-SAME: (%[[INPUT_TUPLE:arg0]]: tuple<tensor<64x128xf32>> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_positions = #ttcore.original_arg_positions<[0]>})
func.func @test_default_type(
    %arg0: tensor<64x128xf32> {ttcore.argument_type = #ttcore.argument_type<default>})
    -> tensor<64x128xf32> {
  // CHECK: %[[GET0:.*]] = ttcore.get_tuple_element %[[INPUT_TUPLE]][0]
  // CHECK: %[[TUPLE_RESULT:.*]] = ttcore.tuple %[[GET0]]
  // CHECK: return %[[TUPLE_RESULT]] : tuple<tensor<64x128xf32>>
  return %arg0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_constant_type
// CHECK-SAME: (%[[PARAM_TUPLE:.*]]: tuple<tensor<64x128xf32>> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.original_arg_positions = #ttcore.original_arg_positions<[0]>})
func.func @test_constant_type(
    %arg0: tensor<64x128xf32> {ttcore.argument_type = #ttcore.argument_type<constant>})
    -> tensor<64x128xf32> {
  // CHECK: %[[GET0:.*]] = ttcore.get_tuple_element %[[PARAM_TUPLE]][0]
  // CHECK: %[[TUPLE_RESULT:.*]] = ttcore.tuple %[[GET0]]
  // CHECK: return %[[TUPLE_RESULT]] : tuple<tensor<64x128xf32>>
  return %arg0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_const_eval
// CHECK-SAME: (%[[SINGLE_TUPLE:arg0]]: tuple<tensor<64x128xf32>, tensor<128x256xf32>>)
func.func @test_const_eval(
    %arg0: tensor<64x128xf32>,
    %arg1: tensor<128x256xf32> )
    -> tensor<64x256xf32> attributes {const_eval} {
  // CHECK: %[[GET0:.*]] = ttcore.get_tuple_element %[[SINGLE_TUPLE]][0]
  // CHECK: %[[GET1:.*]] = ttcore.get_tuple_element %[[SINGLE_TUPLE]][1]
  // CHECK: %[[MATMUL:.*]] = "ttnn.matmul"(%[[GET0]], %[[GET1]])
  %1 = "ttnn.matmul"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<128x256xf32>)-> tensor<64x256xf32>
  // CHECK: %[[TUPLE_RESULT:.*]] = ttcore.tuple %[[MATMUL]] : tuple<tensor<64x256xf32>>
  // CHECK: return %[[TUPLE_RESULT]] : tuple<tensor<64x256xf32>>
  return %1 : tensor<64x256xf32>
}

// Test private functions
// CHECK-LABEL: func.func private @private_func
// CHECK-SAME: (%arg0: tuple<tensor<64x128xf32>> {{.*}})
func.func private @private_func(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  return %arg0 : tensor<64x128xf32>
}

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
// Test target module option
// TARGET-MODULE-CHECK-LABEL: func.func @test_target_module
// TARGET-MODULE-CHECK-SAME: (%arg0: tuple<>)
func.func @test_target_module() -> tensor<64x128xf32, #layout> {
  %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
  // TARGET-MODULE-CHECK: %[[EMPTY_OP:.*]] = "ttnn.empty"
  %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<64x128>}> : (!ttnn.device) -> tensor<64x128xf32, #layout>
  // TARGET-MODULE-CHECK: %[[TUPLE_RESULT:.*]] = ttcore.tuple %[[EMPTY_OP]] : tuple<tensor<64x128xf32, #ttnn_layout>>
  // TARGET-MODULE-CHECK: return %[[TUPLE_RESULT]]
  return %1 : tensor<64x128xf32, #layout>
}

// Test mixed input and parameter ordering
// CHECK-LABEL: func.func @test_mixed_ordering
// CHECK-SAME: (%[[INPUT_TUPLE:arg0]]: tuple<tensor<64x128xf32>, tensor<64x128xf32>> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_positions = #ttcore.original_arg_positions<[0, 2]>},
// CHECK-SAME:  %[[PARAM_TUPLE:arg1]]: tuple<tensor<128x256xf32>, tensor<128x256xf32>> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.original_arg_positions = #ttcore.original_arg_positions<[1, 3]>})
func.func @test_mixed_ordering(
    %arg0: tensor<64x128xf32> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg1: tensor<128x256xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg2: tensor<64x128xf32> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg3: tensor<128x256xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>})
    -> (tensor<128x256xf32>, tensor<64x128xf32>)  {
  // CHECK: %[[GET0:.*]] = ttcore.get_tuple_element %[[INPUT_TUPLE]][0]
  // CHECK: %[[GET2:.*]] = ttcore.get_tuple_element %[[INPUT_TUPLE]][1]
  // CHECK: %[[GET1:.*]] = ttcore.get_tuple_element %[[PARAM_TUPLE]][0]
  // CHECK: %[[GET3:.*]] = ttcore.get_tuple_element %[[PARAM_TUPLE]][1]
  // CHECK: %[[TUPLE_RESULT:.*]] = ttcore.tuple %[[GET1]], %[[GET2]]
  return %arg1, %arg2 : tensor<128x256xf32>, tensor<64x128xf32>
}
