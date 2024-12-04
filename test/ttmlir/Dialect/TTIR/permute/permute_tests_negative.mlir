// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for permute operation

// Verfiy that given attribute `permutation` is a valid permutation of the dimensions.
module {
  func.func @permute_non_valid_permutation(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttir.permute' op Expected a permutation of (0, 1, 2), got (0, 1, 0)
    %0 = tensor.empty() : tensor<16x32x64xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 1, 0>}> : (tensor<16x32x64xbf16>, tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %1 : tensor<16x32x64xbf16>
  }
}

// -----
module {
  func.func @permute_subset_permutation(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttir.permute' op Expected a permutation of (0, 1, 2), got (0, 1)
    %0 = tensor.empty() : tensor<16x32x64xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 1>}> : (tensor<16x32x64xbf16>, tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %1 : tensor<16x32x64xbf16>
  }
}

// Verify that the result shape matches the shape of the input tensor after permutation is applied.
// -----
module {
  func.func @permute_non_valid_shape(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttir.permute' op Expected result shape (16, 64, 32), got (16, 32, 64)
    %0 = tensor.empty() : tensor<16x32x64xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1>}> : (tensor<16x32x64xbf16>, tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %1 : tensor<16x32x64xbf16>
  }
}

// Verify that the result shape matches the shape of the input tensor after permutation is applied.
// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @permute_non_valid_shape(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttir.permute' op Expected result shape (16, 64, 32), got (16, 32, 64)
    %0 = tensor.empty() : tensor<16x32x64xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{operand_constraints = [#any_device, #any_device], permutation = array<i64: 0, 2, 1>}> : (tensor<16x32x64xbf16>, tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %1 : tensor<16x32x64xbf16>
  }
}
