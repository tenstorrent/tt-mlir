// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for matmul operation

// Verfiy that given attribute `permutation` is a valid permutation of the dimensions.
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @permute_non_valid_permutation(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttnn.permute' op Expected a permutation of {k | 0 <= k < 3} got (0, 1, 0)
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i32: 0, 1, 0>}> : (tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %0 : tensor<16x32x64xbf16>
  }
}

// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @permute_non_valid_permutation(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttnn.permute' op Expected a permutation of {k | 0 <= k < 3} got (0, 1)
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i32: 0, 1>}> : (tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %0 : tensor<16x32x64xbf16>
  }
}

// Verify that the result shape matches the shape of the input tensor after permutation is applied.
// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @permute_non_valid_permutation(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttnn.permute' op Expected result shape (16, 64, 32), got (16, 32, 64)
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i32: 0, 2, 1>}> : (tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %0 : tensor<16x32x64xbf16>
  }
}
