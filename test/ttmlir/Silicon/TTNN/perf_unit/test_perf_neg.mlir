// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>

func.func @negate(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  // CHECK: %[[C:.*]] = "ttnn.neg"[[C:.*]]
  %1 = "ttir.neg"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
