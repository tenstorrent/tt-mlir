// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false enable-ttnn-decomposition-pass=false system-desc-path=%system_desc_path%" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func public @dit_rms_norm_unary_fused(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = "ttir.dit_rms_norm_unary_fused"(%arg0, %arg1) <{normalized_shape = array<i64: 4>, epsilon = 1.000000e-05 : f32, activation = "silu", operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
