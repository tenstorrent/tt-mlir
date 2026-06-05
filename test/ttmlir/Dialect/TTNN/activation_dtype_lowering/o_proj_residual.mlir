// RUN: ttmlir-opt --ttnn-ccl-activation-dtype-lowering %s | FileCheck %s

// Projection + residual add: O-projection matmul ->
// [view / reduce_scatter / all_gather]* -> residual add. The matmul's
// result encoding is rewritten to bfp_bf8 (propagating through the CCL
// chain), and the residual add's result encoding stays bf16, so the runtime
// produces bf16 at the block output by combining the bfp_bf8 CCL output
// with the bf16 residual. No explicit ttnn.typecast is emitted.

#dram = #ttnn.buffer_type<dram>
#bf16_2d_a = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_2d_b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_2d_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_2d_scat = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  // CHECK-LABEL: func.func @o_proj_residual
  func.func @o_proj_residual(
      %act:      tensor<32x128xbf16, #bf16_2d_a>,
      %weight:   tensor<128x256xbf16, #bf16_2d_b>,
      %residual: tensor<32x256xbf16, #bf16_2d_out>
  ) -> tensor<32x256xbf16, #bf16_2d_out> {

    // CHECK: "ttnn.matmul"
    // CHECK-SAME: bfp_bf8
    // CHECK-NOT: ttnn.typecast
    %m = "ttnn.matmul"(%act, %weight) <{transpose_a = false, transpose_b = false}>
        : (tensor<32x128xbf16, #bf16_2d_a>, tensor<128x256xbf16, #bf16_2d_b>)
       -> tensor<32x256xbf16, #bf16_2d_out>

    // CHECK: "ttnn.reduce_scatter"
    // CHECK-SAME: bfp_bf8
    // CHECK-NOT: ttnn.typecast
    %rs = "ttnn.reduce_scatter"(%m) <{
        cluster_axis = 1 : ui32,
        reduce_type = #ttcore.reduce_type<sum>,
        scatter_dim = 1 : si32}>
        : (tensor<32x256xbf16, #bf16_2d_out>) -> tensor<32x64xbf16, #bf16_2d_scat>

    // CHECK: "ttnn.all_gather"
    // CHECK-SAME: bfp_bf8
    // CHECK-NOT: ttnn.typecast
    %ag = "ttnn.all_gather"(%rs) <{
        all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}>
        : (tensor<32x64xbf16, #bf16_2d_scat>) -> tensor<32x256xbf16, #bf16_2d_out>

    // CHECK: "ttnn.add"
    // CHECK-NOT: ttnn.typecast
    %out = "ttnn.add"(%residual, %ag)
        : (tensor<32x256xbf16, #bf16_2d_out>, tensor<32x256xbf16, #bf16_2d_out>)
       -> tensor<32x256xbf16, #bf16_2d_out>

    return %out : tensor<32x256xbf16, #bf16_2d_out>
  }
}
