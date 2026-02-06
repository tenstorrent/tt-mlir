// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func @group_norm(%arg0: tensor<1x1x64x480xbf16>, %arg1: tensor<1x8x32x64xbf16>, %arg2: tensor<480xbf16>, %arg3: tensor<480xbf16>) -> tensor<1x1x64x480xbf16> {
    %1 = "ttir.group_norm"(%arg0, %arg1, %arg2, %arg3) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<1x1x64x480xbf16>, tensor<1x8x32x64xbf16>, tensor<480xbf16>, tensor<480xbf16>) -> tensor<1x1x64x480xbf16>
    return %1 : tensor<1x1x64x480xbf16>
  }
}
