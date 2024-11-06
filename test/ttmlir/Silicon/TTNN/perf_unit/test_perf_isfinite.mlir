// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>

func.func @is_finite(%arg0: tensor<64x128xf32>) -> tensor<64x128xbf16> {
  // CHECK: %[[C:.*]] = "ttnn.empty"
  // CHECK-SAME: [[TENSOR:tensor<64x128xbf16,]]
  %0 = tensor.empty() : tensor<64x128xbf16>
  // CHECK: %[[C:.*]] = "ttnn.isfinite"
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: [[TENSOR]]
  // CHECK-SAME: -> [[TENSOR]]
  %1 = "ttir.isfinite"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}
