// RUN: ttmlir-opt --ttir-to-ttmetal-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>

func.func @maximum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.maximum"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
