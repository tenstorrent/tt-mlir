// RUN: ttmlir-opt --ttnn-canonicalize-function-arguments %s | FileCheck %s
//
// Test for --ttnn-canonicalize-function-arguments pass.
// The pass should annotate function arguments with their original positions and assign
// #ttcore.argument_type<input> to all function arguments that doesn't have argument type assigned.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
module {
  // CHECK-LABEL: func.func @no_attributes
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<0>},
  // CHECK-SAME:  %arg1: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<1>},
  // CHECK-SAME:  %arg2: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<2>})
  func.func @no_attributes(
    %arg0: tensor<32x32xbf16, #layout>,
    %arg1: tensor<32x32xbf16, #layout>,
    %arg2: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    %1 = "ttnn.multiply"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    return %1 : tensor<32x32xbf16, #layout>
  }

  // CHECK-LABEL: func.func @mixed_attributes
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<0>},
  // CHECK-SAME:  %arg1: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.original_arg_position = #ttcore.original_arg_position<1>},
  // CHECK-SAME:  %arg2: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<2>},
  // CHECK-SAME:  %arg3: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.original_arg_position = #ttcore.original_arg_position<3>})
  func.func @mixed_attributes(
    %arg0: tensor<32x32xbf16, #layout>,
    %arg1: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg2: tensor<32x32xbf16, #layout>,
    %arg3: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16, #layout> {
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    %1 = "ttnn.subtract"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    %2 = "ttnn.multiply"(%1, %arg3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    return %2 : tensor<32x32xbf16, #layout>
  }

  // CHECK-LABEL: func.func @already_has_attributes
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<0>},
  // CHECK-SAME:  %arg1: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.original_arg_position = #ttcore.original_arg_position<1>},
  // CHECK-SAME:  %arg2: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.original_arg_position = #ttcore.original_arg_position<2>})
  func.func @already_has_attributes(
    %arg0: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg1: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg2: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16, #layout> {
    %0 = "ttnn.multiply"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    %1 = "ttnn.add"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    return %1 : tensor<32x32xbf16, #layout>
  }

  // CHECK-LABEL: func.func @preserve_other_attributes
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<0>, ttir.name = "input_tensor"},
  // CHECK-SAME:  %arg1: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<1>, ttir.name = "weight_tensor"})
  func.func @preserve_other_attributes(
    %arg0: tensor<32x32xbf16, #layout> {ttir.name = "input_tensor"},
    %arg1: tensor<32x32xbf16, #layout> {ttir.name = "weight_tensor"}) -> tensor<32x32xbf16, #layout> {
    %0 = "ttnn.multiply"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    return %0 : tensor<32x32xbf16, #layout>
  }

  // CHECK-LABEL: func.func @no_args
  func.func @no_args() -> tensor<32x32xbf16, #layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
    %2 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
    %3 = "ttnn.multiply"(%1, %2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    return %3 : tensor<32x32xbf16, #layout>
  }
}
