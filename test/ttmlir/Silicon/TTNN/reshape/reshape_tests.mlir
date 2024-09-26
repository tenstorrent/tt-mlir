// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>

func.func @reshape1(%arg0: tensor<4x2x32x34xbf16>) -> tensor<2x4x32x34xbf16> {
  %0 = tensor.empty() : tensor<2x4x32x34xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 32: i32, 34: i32] , operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<4x2x32x34xbf16>, tensor<2x4x32x34xbf16>) -> tensor<2x4x32x34xbf16>
  return %1 : tensor<2x4x32x34xbf16>
}

func.func @reshape2(%arg0: tensor<3x3x32x64xbf16>) -> tensor<9x32x64xbf16> {
  %0 = tensor.empty() : tensor<9x32x64xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [9: i32, 32: i32, 64: i32], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<3x3x32x64xbf16>, tensor<9x32x64xbf16>) -> tensor<9x32x64xbf16>
  return %1 : tensor<9x32x64xbf16>
}

func.func @reshape3(%arg0: tensor<8x8x16x16xbf16>) -> tensor<64x16x16xbf16> {
  %0 = tensor.empty() : tensor<64x16x16xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [64: i32, 16: i32, 16: i32], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<8x8x16x16xbf16>, tensor<64x16x16xbf16>) -> tensor<64x16x16xbf16>
  return %1 : tensor<64x16x16xbf16>
}

func.func @reshape4(%arg0: tensor<6x4x64x64xbf16>) -> tensor<24x64x64xbf16> {
  %0 = tensor.empty() : tensor<24x64x64xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [24: i32, 64: i32, 64: i32], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<6x4x64x64xbf16>, tensor<24x64x64xbf16>) -> tensor<24x64x64xbf16>
  return %1 : tensor<24x64x64xbf16>
}

func.func @reshape5(%arg0: tensor<2x2x16x32xbf16>) -> tensor<4x16x32xbf16> {
  %0 = tensor.empty() : tensor<4x16x32xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [4: i32, 16: i32, 32: i32], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<2x2x16x32xbf16>, tensor<4x16x32xbf16>) -> tensor<4x16x32xbf16>
  return %1 : tensor<4x16x32xbf16>
}

func.func @reshape6(%arg0: tensor<4x4x32x32xbf16>) -> tensor<16x32x32xbf16> {
  %0 = tensor.empty() : tensor<16x32x32xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [16: i32, 32: i32, 32: i32], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<4x4x32x32xbf16>, tensor<16x32x32xbf16>) -> tensor<16x32x32xbf16>
  return %1 : tensor<16x32x32xbf16>
}

func.func @reshape7(%arg0: tensor<33x24x62xbf16>) -> tensor<24x33x62xbf16> {
  %0 = tensor.empty() : tensor<24x33x62xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [24: i32, 33: i32, 62: i32], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<33x24x62xbf16>, tensor<24x33x62xbf16>) -> tensor<24x33x62xbf16>
  return %1 : tensor<24x33x62xbf16>
}

func.func @reshape_tile_aligned1(%arg0: tensor<8x4x32x32xbf16>) -> tensor<32x32x32xbf16> {
  %0 = tensor.empty() : tensor<32x32x32xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [32: i32, 32: i32, 32: i32], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<8x4x32x32xbf16>, tensor<32x32x32xbf16>) -> tensor<32x32x32xbf16>
  return %1 : tensor<32x32x32xbf16>
}

func.func @reshape_tile_aligned2(%arg0: tensor<16x2x32x64xbf16>) -> tensor<32x32x64xbf16> {
  %0 = tensor.empty() : tensor<32x32x64xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [32: i32, 32: i32, 64: i32], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<16x2x32x64xbf16>, tensor<32x32x64xbf16>) -> tensor<32x32x64xbf16>
  return %1 : tensor<32x32x64xbf16>
}
