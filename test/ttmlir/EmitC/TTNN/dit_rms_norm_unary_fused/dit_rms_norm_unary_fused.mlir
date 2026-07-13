// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path% optimization-level=1" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

// The fused op is produced by the ttnn-fusing pass from rms_norm + silu (only
// under the op-model-gated optimizer path); this test exercises the
// TTNN -> EmitC / flatbuffer path for it.
module {
  func.func public @dit_rms_norm_unary_fused(%arg0: tensor<32x512xf32>, %arg1: tensor<512xf32>) -> tensor<32x512xf32> {
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{normalized_shape = array<i64: 512>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x512xf32>, tensor<512xf32>) -> tensor<32x512xf32>
    %1 = "ttir.silu"(%0) : (tensor<32x512xf32>) -> tensor<32x512xf32>
    return %1 : tensor<32x512xf32>
  }
}
