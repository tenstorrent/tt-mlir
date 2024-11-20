// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>

module attributes{} {
  func.func @add(
    %arg0: tensor<32x32xf32>,  // First input tensor
    %arg1: tensor<32x32xf32>,  // Second input tensor
    %arg2: tensor<32x32xf32>   // Output tensor (result stored here)
  ) -> tensor<32x32xf32> {
    %1 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
     // CHECK: [[VAL1:%[0-9]+]] = "linalg.add"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, %arg{{[0-9]+}}){{.+}}: ([[TENSOR_SIZE]], [[TENSOR_SIZE]], [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    //    %1 = linalg.add ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
