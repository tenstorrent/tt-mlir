// RUN: ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for matmul operation

// Verify that the parsing fails if either of operands is a scalar
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @simple_maximum_example(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    // CHECK: error: 'ttir.matmul' op Input A must be at least a 1D tensor
    %0 = tensor.empty() : tensor<bf16>
    %1 = "ttir.maximum"(%arg0, %arg1, %0) <{operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<bf16>, tensor<bf16>, tensor<bf16>) -> tensor<bf16>
    return %1 : tensor<bf16>
  }
}
