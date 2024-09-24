// REQUIRES: wormhole_b0
// REQUIRES: functional,perf
// REQUIRES: n150,n300
// REQUIRES: push
// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-to-ttmetal-backend-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>

func.func public @add5(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttmetal.host_write"[[C:.*]]
  %0 = "ttir.constant"() <{value = dense<5.0> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  %1 = tensor.empty() : tensor<32x32xf32>
  %2 = "ttir.add"(%arg0, %0, %1) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}
