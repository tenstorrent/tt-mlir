// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for matmul operation

// Verify that the parsing fails if input and output shapes do not match.
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @clamp(%arg0: tensor<64x64xbf16>) -> tensor<64x128xbf16> {
    // CHECK: error: 'ttnn.clamp' op input and output must have same shape.
    %0 = tensor.empty() : tensor<64x128xbf16>
    %1 = "ttnn.clamp"(%arg0) <{max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}> : (tensor<64x64xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}

// Verify that parsing fails in case of more than one input.
// -----
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @clamp2(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: error: 'ttnn.clamp' op expects one tensor as input.
    %0 = tensor.empty() : tensor<64x128xbf16>
    %1 = "ttnn.clamp"(%arg0, %arg1) <{max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
