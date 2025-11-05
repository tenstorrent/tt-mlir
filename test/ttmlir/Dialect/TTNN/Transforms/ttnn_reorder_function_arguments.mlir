// RUN: ttmlir-opt --ttnn-canonicalize-function-arguments --ttnn-reorder-function-arguments %s | FileCheck %s
//
// Test for --ttnn-reorder-function-arguments pass.
// The pass should reorder function arguments so that all inputs come first, followed by parameters and constants.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
module {
  // CHECK-LABEL: func.func @mixed_args
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<0>},
  // CHECK-SAME:  %arg1: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<2>},
  // CHECK-SAME:  %arg2: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.original_arg_position = #ttcore.original_arg_position<1>},
  // CHECK-SAME:  %arg3: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.original_arg_position = #ttcore.original_arg_position<3>})
  func.func @mixed_args(
    %arg0: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg1: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg2: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg3: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16, #layout> {
    // CHECK: "ttnn.add"(%arg0, %arg2)
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    // CHECK: "ttnn.subtract"(%0, %arg1)
    %1 = "ttnn.subtract"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    // CHECK: "ttnn.multiply"(%1, %arg3)
    %2 = "ttnn.multiply"(%1, %arg3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    return %2 : tensor<32x32xbf16, #layout>
  }

  // CHECK-LABEL: func.func @already_ordered
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<0>},
  // CHECK-SAME:  %arg1: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.original_arg_position = #ttcore.original_arg_position<1>},
  // CHECK-SAME:  %arg2: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.original_arg_position = #ttcore.original_arg_position<2>})
  func.func @already_ordered(
    %arg0: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg1: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg2: tensor<32x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16, #layout> {
    // CHECK: "ttnn.multiply"(%arg0, %arg1)
    %0 = "ttnn.multiply"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    // CHECK: "ttnn.add"(%0, %arg2)
    %1 = "ttnn.add"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    return %1 : tensor<32x32xbf16, #layout>
  }

  // CHECK-LABEL: func.func @only_inputs
  // CHECK-SAME: (%arg0: tensor<32x16xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<0>},
  // CHECK-SAME:  %arg1: tensor<32x16xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<1>},
  // CHECK-SAME:  %arg2: tensor<16x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_position = #ttcore.original_arg_position<2>})
  func.func @only_inputs(
    %arg0: tensor<32x16xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg1: tensor<32x16xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg2: tensor<16x32xbf16, #layout> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<16x32xbf16, #layout> {
    // CHECK: "ttnn.subtract"(%arg0, %arg1)
    %0 = "ttnn.subtract"(%arg0, %arg1)  <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x16xbf16, #layout>, tensor<32x16xbf16, #layout>) -> tensor<32x16xbf16, #layout>
    // CHECK: "ttnn.reshape"(%0)
    %1 = "ttnn.reshape"(%0) <{shape = [16: i32, 32: i32]}> : (tensor<32x16xbf16, #layout>) -> tensor<16x32xbf16, #layout>
    // CHECK: "ttnn.add"(%1, %arg2)
    %5 = "ttnn.add"(%1, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<16x32xbf16, #layout>, tensor<16x32xbf16, #layout>) -> tensor<16x32xbf16, #layout>
    return %5 : tensor<16x32xbf16, #layout>
  }
}
