// RUN: ttmlir-opt --ttnn-activation-dtype-lowering=enable=true %s | FileCheck %s

// Pattern A: QKV matmul -> [view / CCL]* -> rotary_embedding.
// The producer matmul's output dtype is lowered to bfp_bf8 and propagates
// through the CCL chain. A ttnn.typecast back to bf16 is inserted before
// rotary_embedding (which requires bf16).

#dram = #ttnn.buffer_type<dram>
#bf16_a    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_b    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_out  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_scat = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_4d   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_cs   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  // CHECK-LABEL: func.func @qkv_rope
  func.func @qkv_rope(
      %act:    tensor<32x128xbf16, #bf16_a>,
      %weight: tensor<128x128xbf16, #bf16_b>,
      %cos:    tensor<1x1x32x128xbf16, #bf16_cs>,
      %sin:    tensor<1x1x32x128xbf16, #bf16_cs>
  ) -> tensor<1x1x32x128xbf16, #bf16_4d> {

    // CHECK: "ttnn.matmul"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    // CHECK-SAME: bfp_bf8
    %m = "ttnn.matmul"(%act, %weight) <{transpose_a = false, transpose_b = false}>
        : (tensor<32x128xbf16, #bf16_a>, tensor<128x128xbf16, #bf16_b>)
       -> tensor<32x128xbf16, #bf16_out>

    // CHECK: "ttnn.reduce_scatter"
    // CHECK-SAME: bfp_bf8
    %rs = "ttnn.reduce_scatter"(%m) <{
        cluster_axis = 0 : ui32,
        reduce_type = #ttcore.reduce_type<sum>,
        scatter_dim = 1 : si32}>
        : (tensor<32x128xbf16, #bf16_out>) -> tensor<32x32xbf16, #bf16_scat>

    // CHECK: "ttnn.all_gather"
    // CHECK-SAME: bfp_bf8
    %ag = "ttnn.all_gather"(%rs) <{
        all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}>
        : (tensor<32x32xbf16, #bf16_scat>) -> tensor<32x128xbf16, #bf16_out>

    // CHECK: "ttnn.reshape"
    // CHECK-SAME: bfp_bf8
    %r = "ttnn.reshape"(%ag) <{shape = [1 : i32, 1 : i32, 32 : i32, 128 : i32]}>
        : (tensor<32x128xbf16, #bf16_out>) -> tensor<1x1x32x128xbf16, #bf16_4d>

    // The pass must insert a ttnn.typecast to bf16 before the RoPE consumer.
    // CHECK: %[[CAST:.*]] = "ttnn.typecast"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.rotary_embedding"(%[[CAST]],
    %out = "ttnn.rotary_embedding"(%r, %cos, %sin) <{token_index = 0 : ui32}>
        : (tensor<1x1x32x128xbf16, #bf16_4d>,
           tensor<1x1x32x128xbf16, #bf16_cs>,
           tensor<1x1x32x128xbf16, #bf16_cs>) -> tensor<1x1x32x128xbf16, #bf16_4d>

    return %out : tensor<1x1x32x128xbf16, #bf16_4d>
  }
}
