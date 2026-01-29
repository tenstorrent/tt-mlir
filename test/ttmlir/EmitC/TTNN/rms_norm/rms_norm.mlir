// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func public @rms_norm(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{normalized_shape = array<i64: 4>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
