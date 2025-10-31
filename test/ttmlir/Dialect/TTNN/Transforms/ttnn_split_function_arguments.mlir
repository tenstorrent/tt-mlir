// RUN: ttmlir-opt --ttnn-split-function-arguments %s | FileCheck %s
//
// Test for --ttnn-split-function-arguments pass.
// The pass should reorder function arguments so that all inputs come first, followed by parameters and constants.

module {
  // CHECK-LABEL: func.func @mixed_args
  // CHECK-SAME: (%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_num = 0 : i64},
  // CHECK-SAME:  %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.original_arg_num = 2 : i64},
  // CHECK-SAME:  %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.original_arg_num = 1 : i64},
  // CHECK-SAME:  %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.original_arg_num = 3 : i64})
  func.func @mixed_args(
    %arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.add"(%arg0, %arg2, %0)
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.subtract"(%1, %arg1, %2)
    %3 = "ttir.subtract"(%1, %arg2, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttir.multiply"(%3, %arg3, %4)
    %5 = "ttir.multiply"(%3, %arg3, %4) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %5 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func @already_ordered
  // CHECK-SAME: (%arg0: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
  // CHECK-SAME:  %arg1: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>})
  func.func @already_ordered(
    %arg0: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg1: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<64x64xbf16> {
    %0 = ttir.empty() : tensor<64x64xbf16>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  // CHECK-LABEL: func.func @only_inputs
  // CHECK-SAME: (%arg0: tensor<32x16xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
  // CHECK-SAME:  %arg1: tensor<32x16xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
  // CHECK-SAME:  %arg2: tensor<16x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>})
  func.func @only_inputs(
    %arg0: tensor<32x16xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg1: tensor<32x16xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg2: tensor<16x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<16x32xbf16> {
    %0 = ttir.empty() : tensor<32x16xbf16>
    %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<32x16xbf16>, tensor<32x16xbf16>, tensor<32x16xbf16>) -> tensor<32x16xbf16>
    %2 = ttir.empty() : tensor<16x32xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [16: i32, 32: i32]}> : (tensor<32x16xbf16>, tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %4 = ttir.empty() : tensor<16x32xbf16>
    %5 = "ttir.add"(%arg2, %3, %4) : (tensor<16x32xbf16>, tensor<16x32xbf16> , tensor<16x32xbf16>) -> tensor<16x32xbf16>
    return %5 : tensor<16x32xbf16>
  }
}
