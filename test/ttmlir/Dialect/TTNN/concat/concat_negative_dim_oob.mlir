// RUN: not ttmlir-opt --ttir-to-ttnn-backend-pipeline %s 2>&1 | FileCheck %s
// CHECK: error: 'ttir.concat' op Invalid dimension -3 for concatenation.
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x96xf32> {
    %0 = tensor.empty() : tensor<32x96xf32>
    %1 = "ttir.concat"(%arg0, %arg1, %0) <{axis = -3 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x96xf32>) -> tensor<32x96xf32>
    return %1 : tensor<32x96xf32>
  }
}
