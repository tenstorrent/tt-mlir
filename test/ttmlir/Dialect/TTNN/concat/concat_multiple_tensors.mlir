// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward() -> tensor<32x224xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<32x32xf32>
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %1 = tensor.empty() : tensor<32x64xf32>
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %2 = tensor.empty() : tensor<32x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %3 = tensor.empty() : tensor<32x224xf32>
    // CHECK: %[[C:.*]] = "ttnn.concat"[[C:.*]]
    %4 = "ttir.concat"(%0, %1, %2, %3) <{dim = 1 : si32, operand_constraints = [#any_device, #any_device, #any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x128xf32>, tensor<32x224xf32>) -> tensor<32x224xf32>
    return %4 : tensor<32x224xf32>
  }
}
