// RUN: ttmlir-opt --ttnn-ccl-activation-dtype-lowering %s | FileCheck %s

// Negative tests for the MLP up/gate matcher's strictness. The matcher only
// fires when the gate multiply feeds *exactly one* consumer and that consumer
// is the FF2 (down) matmul rooting a projection-residual chain. The two cases
// below each violate one of those conditions, so nothing should be lowered to
// bfp_bf8.

#dram = #ttnn.buffer_type<dram>
#bf16_a    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_b    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_b2   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_act  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_scat = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_out  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  // Case 1: the gate multiply fans out to a second consumer (the FF2 matmul
  // *and* an extra add), so it does not have a single use. The matcher must
  // not lower anything.
  // CHECK-LABEL: func.func @mlp_gate_multi_use
  // CHECK-NOT: bfp_bf8
  func.func @mlp_gate_multi_use(
      %x:     tensor<32x128xbf16, #bf16_a>,
      %w_ff1: tensor<128x256xbf16, #bf16_b>,
      %w_ff3: tensor<128x256xbf16, #bf16_b>,
      %w_ff2: tensor<256x128xbf16, #bf16_b2>
  ) -> (tensor<32x128xbf16, #bf16_out>, tensor<32x256xbf16, #bf16_act>) {

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
    %silu = "ttnn.silu"(%ag1) : (tensor<32x256xbf16, #bf16_act>)
                              -> tensor<32x256xbf16, #bf16_act>

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

    %gate = "ttnn.multiply"(%silu, %ag3)
        : (tensor<32x256xbf16, #bf16_act>, tensor<32x256xbf16, #bf16_act>)
       -> tensor<32x256xbf16, #bf16_act>

    // First consumer: the FF2 (down) matmul.
    %ff2 = "ttnn.matmul"(%gate, %w_ff2) <{transpose_a = false, transpose_b = false}>
        : (tensor<32x256xbf16, #bf16_act>, tensor<256x128xbf16, #bf16_b2>)
       -> tensor<32x128xbf16, #bf16_out>
    // Second consumer: extra use of the gate -> multiply is not single-use.
    %extra = "ttnn.add"(%gate, %gate)
        : (tensor<32x256xbf16, #bf16_act>, tensor<32x256xbf16, #bf16_act>)
       -> tensor<32x256xbf16, #bf16_act>

    return %ff2, %extra : tensor<32x128xbf16, #bf16_out>, tensor<32x256xbf16, #bf16_act>
  }

  // Case 2: the gate multiply has a single use, but its consumer is not a
  // matmul (it feeds another silu), so it is not the FF2 down-projection. The
  // matcher must not lower anything.
  // CHECK-LABEL: func.func @mlp_gate_consumer_not_matmul
  // CHECK-NOT: bfp_bf8
  func.func @mlp_gate_consumer_not_matmul(
      %x:     tensor<32x128xbf16, #bf16_a>,
      %w_ff1: tensor<128x256xbf16, #bf16_b>,
      %w_ff3: tensor<128x256xbf16, #bf16_b>
  ) -> tensor<32x256xbf16, #bf16_act> {

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
    %silu = "ttnn.silu"(%ag1) : (tensor<32x256xbf16, #bf16_act>)
                              -> tensor<32x256xbf16, #bf16_act>

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

    %gate = "ttnn.multiply"(%silu, %ag3)
        : (tensor<32x256xbf16, #bf16_act>, tensor<32x256xbf16, #bf16_act>)
       -> tensor<32x256xbf16, #bf16_act>

    // Single use, but the consumer is a silu, not the FF2 matmul.
    %out = "ttnn.silu"(%gate) : (tensor<32x256xbf16, #bf16_act>)
                              -> tensor<32x256xbf16, #bf16_act>

    return %out : tensor<32x256xbf16, #bf16_act>
  }
}
