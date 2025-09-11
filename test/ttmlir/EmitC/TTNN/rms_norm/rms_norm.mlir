// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func public @rms_norm(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = ttir.empty() : tensor<2x4xf32>
    %1 = "ttir.rms_norm"(%arg0, %arg1, %0) <{normalized_shape = array<i64: 4>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 1>}> : (tensor<2x4xf32>, tensor<4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}
