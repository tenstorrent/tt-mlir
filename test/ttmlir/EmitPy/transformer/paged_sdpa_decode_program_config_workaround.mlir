// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --convert-ttnn-to-emitpy -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python %t.mlir | FileCheck %s

func.func @test_paged_sdpa_emit_program_config(
    %arg0: tensor<1x1x4x512xbf16>,
    %arg1: tensor<128x4x32x512xbf16>,
    %arg2: tensor<128x4x32x512xbf16>,
    %arg3: tensor<1x64xi32>,
    %arg4: tensor<1xi32>) -> tensor<1x1x4x512xbf16> {
  // CHECK: ttnn.SDPAProgramConfig(
  // CHECK-SAME: compute_with_storage_grid_size=ttnn.CoreCoord(
  // CHECK-SAME: q_chunk_size=
  // CHECK-SAME: k_chunk_size=32
  // CHECK-SAME: max_cores_per_head_batch=32
  %result = "ttnn.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %arg4)
      <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>}>
      : (tensor<1x1x4x512xbf16>, tensor<128x4x32x512xbf16>, tensor<128x4x32x512xbf16>,
         tensor<1x64xi32>, tensor<1xi32>) -> tensor<1x1x4x512xbf16>
  return %result : tensor<1x1x4x512xbf16>
}
