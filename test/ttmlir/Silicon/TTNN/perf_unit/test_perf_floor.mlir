// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>

func.func @floor(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %{{[0-9]+}} = "ttnn.empty"
  // CHECK-SAME: [[TENSOR:tensor<64x128xf32,]]
  %0 = tensor.empty() : tensor<64x128xf32>
  // CHECK: %{{[0-9]+}} = "ttnn.floor"
  // CHECK-SAME: [[TENSOR]]
  // CHECK-SAME: [[TENSOR]]
  // CHECK-SAME: -> [[TENSOR]]
  %1 = "ttir.floor"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
