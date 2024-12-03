// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for matmul operation

// Verfiy that given attribute `permutation` is a valid permutation of the dimensions
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @permute_non_valid_permutation(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttir.permute' op Invalid permutation
    %0 = tensor.empty() : tensor<16x32x64xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{operand_constraints = [#any_device, #any_device], permutation = array<i32: 0, 1, 0>}> : (tensor<16x32x64xbf16>, tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %1 : tensor<16x32x64xbf16>
  }
}

// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @permute_non_valid_permutation(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttir.permute' op Invalid permutation
    %0 = tensor.empty() : tensor<16x32x64xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{operand_constraints = [#any_device, #any_device], permutation = array<i32: 0, 1>}> : (tensor<16x32x64xbf16>, tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %1 : tensor<16x32x64xbf16>
  }
}
