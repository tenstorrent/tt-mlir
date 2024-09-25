// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>

func.func @forward(%arg0: tensor<4x2x32x32xbf16>, %arg1: tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  %0 = tensor.empty() : tensor<2x4x32x32xbf16>
  // Reshape the first tensor from shape 4x2x32x32 to 2x4x32x32
  // CHECK: %[[C1:.*]] = "ttnn.reshape"[[C1:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 32: i32, 32: i32], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<4x2x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>

  %2 = tensor.empty() : tensor<2x4x32x32xbf16>
  // Add the reshaped tensor %1 to %arg1
  // CHECK: %[[C2:.*]] = "ttnn.add"[[C2:.*]]
  %3 = "ttir.add"(%1, %arg1, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<2x4x32x32xbf16>, tensor<2x4x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>

  return %3 : tensor<2x4x32x32xbf16>
}
