// RUN: ttmlir-opt --ttnn-ccl-activation-dtype-lowering %s | FileCheck %s

// MLP FF1/FF2/FF3 + residual add: FF1 (up) and FF3 (gate) matmuls ->
// [view / CCL]* -> silu / multiply. The multiply feeds FF2 (down), and
// FF2's matmul -> [view / CCL]* -> residual add restores bf16 at the
// block output.
//
// The pass should:
//   - Rewrite FF1, FF3, FF2 matmul result encodings to bfp_bf8.
//   - Propagate bfp_bf8 through their CCL chains.
//   - Rewrite the gate multiply's result encoding to bfp_bf8.
//   - Leave the final residual add's result encoding as bf16 (TTNN derives
//     output dtype from the result encoding, restoring bf16 at the exit).

#dram = #ttnn.buffer_type<dram>
#bf16_a    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_b    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_b2   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_act  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_scat = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_out  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_out_scat = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  // CHECK-LABEL: func.func @mlp_123_residual
  func.func @mlp_123_residual(
      %x:        tensor<32x128xbf16, #bf16_a>,
      %w_ff1:    tensor<128x256xbf16, #bf16_b>,
      %w_ff3:    tensor<128x256xbf16, #bf16_b>,
      %w_ff2:    tensor<256x128xbf16, #bf16_b2>,
      %residual: tensor<32x128xbf16, #bf16_out>
  ) -> tensor<32x128xbf16, #bf16_out> {

    // FF1 (up) matmul -> CCL -> silu -> multiply (gate)
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: bfp_bf8
    // CHECK-NOT: ttnn.typecast
    %ff1 = "ttnn.matmul"(%x, %w_ff1) <{transpose_a = false, transpose_b = false}>
        : (tensor<32x128xbf16, #bf16_a>, tensor<128x256xbf16, #bf16_b>)
       -> tensor<32x256xbf16, #bf16_act>

    %rs1 = "ttnn.reduce_scatter"(%ff1) <{
        cluster_axis = 0 : ui32,
        reduce_type = #ttcore.reduce_type<sum>,
        scatter_dim = 1 : si32}>
        : (tensor<32x256xbf16, #bf16_act>) -> tensor<32x64xbf16, #bf16_scat>

    %ag1 = "ttnn.all_gather"(%rs1) <{
        all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}>
        : (tensor<32x64xbf16, #bf16_scat>) -> tensor<32x256xbf16, #bf16_act>

    // CHECK: "ttnn.silu"
    // CHECK-SAME: bfp_bf8
    // CHECK-NOT: ttnn.typecast
    %silu = "ttnn.silu"(%ag1) : (tensor<32x256xbf16, #bf16_act>)
                              -> tensor<32x256xbf16, #bf16_act>

    // FF3 (gate) matmul -> CCL -> multiply
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: bfp_bf8
    // CHECK-NOT: ttnn.typecast
    %ff3 = "ttnn.matmul"(%x, %w_ff3) <{transpose_a = false, transpose_b = false}>
        : (tensor<32x128xbf16, #bf16_a>, tensor<128x256xbf16, #bf16_b>)
       -> tensor<32x256xbf16, #bf16_act>

    %rs3 = "ttnn.reduce_scatter"(%ff3) <{
        cluster_axis = 0 : ui32,
        reduce_type = #ttcore.reduce_type<sum>,
        scatter_dim = 1 : si32}>
        : (tensor<32x256xbf16, #bf16_act>) -> tensor<32x64xbf16, #bf16_scat>

    %ag3 = "ttnn.all_gather"(%rs3) <{
        all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}>
        : (tensor<32x64xbf16, #bf16_scat>) -> tensor<32x256xbf16, #bf16_act>

    // CHECK: "ttnn.multiply"
    // CHECK-SAME: bfp_bf8
    // CHECK-NOT: ttnn.typecast
    %gate = "ttnn.multiply"(%silu, %ag3)
        : (tensor<32x256xbf16, #bf16_act>, tensor<32x256xbf16, #bf16_act>)
       -> tensor<32x256xbf16, #bf16_act>

    // FF2 (down) matmul -> CCL -> add (residual). Last add must restore bf16.
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: bfp_bf8
    // CHECK-NOT: ttnn.typecast
    %ff2 = "ttnn.matmul"(%gate, %w_ff2) <{transpose_a = false, transpose_b = false}>
        : (tensor<32x256xbf16, #bf16_act>, tensor<256x128xbf16, #bf16_b2>)
       -> tensor<32x128xbf16, #bf16_out>

    %rs2 = "ttnn.reduce_scatter"(%ff2) <{
        cluster_axis = 1 : ui32,
        reduce_type = #ttcore.reduce_type<sum>,
        scatter_dim = 1 : si32}>
        : (tensor<32x128xbf16, #bf16_out>) -> tensor<32x32xbf16, #bf16_out_scat>

    %ag2 = "ttnn.all_gather"(%rs2) <{
        all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}>
        : (tensor<32x32xbf16, #bf16_out_scat>) -> tensor<32x128xbf16, #bf16_out>

    // CHECK: "ttnn.add"
    // CHECK-NOT: ttnn.typecast
    %out = "ttnn.add"(%residual, %ag2)
        : (tensor<32x128xbf16, #bf16_out>, tensor<32x128xbf16, #bf16_out>)
       -> tensor<32x128xbf16, #bf16_out>

    return %out : tensor<32x128xbf16, #bf16_out>
  }
}
