// RUN: ttmlir-opt --ttir-constant-as-fill %s | FileCheck %s

#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>

func.func public @add5(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
  // CHECK: %[[C:.*]] = "ttir.fill"[[C:.*]]
  %0 = "ttir.constant"() <{value = dense<5.000000e+00> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
  %1 = tensor.empty() : tensor<32x32xf32>
  %2 = "ttir.add"(%arg0, %0, %1) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}
