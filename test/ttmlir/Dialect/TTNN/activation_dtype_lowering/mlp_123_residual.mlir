// RUN: ttmlir-opt --ttnn-activation-dtype-lowering=enable=true %s | FileCheck %s

// Pattern C: MLP FF1 (up) and FF3 (gate) matmuls -> [view / CCL]* -> silu /
// multiply, with the multiply feeding FF2. FF2's matmul -> [view / CCL]* ->
// residual add (Pattern B tail).
//
// The pass should:
//   - Lower FF1, FF3, FF2 matmul outputs to bfp_bf8.
//   - Propagate bfp_bf8 through their CCL chains.
//   - Set the gate `multiply.dtype = bfp_bf8` so the bfp8 path continues.
//   - Set the final residual `add.dtype = bf16`.

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
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
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
    %silu = "ttnn.silu"(%ag1) : (tensor<32x256xbf16, #bf16_act>)
                              -> tensor<32x256xbf16, #bf16_act>

    // FF3 (gate) matmul -> CCL -> multiply
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
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    %gate = "ttnn.multiply"(%silu, %ag3)
        : (tensor<32x256xbf16, #bf16_act>, tensor<32x256xbf16, #bf16_act>)
       -> tensor<32x256xbf16, #bf16_act>

    // FF2 (down) matmul -> CCL -> add (residual). Last add must restore bf16.
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
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
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-NOT: ttnn.typecast
    %out = "ttnn.add"(%residual, %ag2)
        : (tensor<32x128xbf16, #bf16_out>, tensor<32x128xbf16, #bf16_out>)
       -> tensor<32x128xbf16, #bf16_out>

    return %out : tensor<32x128xbf16, #bf16_out>
  }
}
