// RUN: ttmlir-opt --ttnn-activation-dtype-lowering=enable=true %s | FileCheck %s

// Pattern D: LM-head matmul -> reshape -> all_gather (dim 0, expanding) ->
// sum (dim 0, collapsing) -> reshape -> all_gather (dim 2) -> argmax.
// The matmul output is lowered to bfp_bf8, propagating through view / CCL /
// sum. A ttnn.typecast back to bf16 is inserted before argmax (which
// requires bf16).

#dram = #ttnn.buffer_type<dram>
#bf16_a    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_b    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_out  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_3d   = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_3d_g = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<4x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_3d_p = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<1x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#u32_out   = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>

module attributes {} {
  // CHECK-LABEL: func.func @lm_head_argmax
  func.func @lm_head_argmax(
      %act:    tensor<32x128xbf16, #bf16_a>,
      %weight: tensor<128x512xbf16, #bf16_b>
  ) -> tensor<1x32x1xui32, #u32_out> {

    // CHECK: "ttnn.matmul"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    %m = "ttnn.matmul"(%act, %weight) <{transpose_a = false, transpose_b = false}>
        : (tensor<32x128xbf16, #bf16_a>, tensor<128x512xbf16, #bf16_b>)
       -> tensor<32x512xbf16, #bf16_out>

    // CHECK: "ttnn.reshape"
    // CHECK-SAME: bfp_bf8
    %r1 = "ttnn.reshape"(%m) <{shape = [1 : i32, 32 : i32, 512 : i32]}>
        : (tensor<32x512xbf16, #bf16_out>) -> tensor<1x32x512xbf16, #bf16_3d>

    // CHECK: "ttnn.all_gather"
    // CHECK-SAME: bfp_bf8
    %ag1 = "ttnn.all_gather"(%r1) <{
        all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}>
        : (tensor<1x32x512xbf16, #bf16_3d>) -> tensor<4x32x512xbf16, #bf16_3d_g>

    // CHECK: "ttnn.sum"
    // CHECK-SAME: bfp_bf8
    %s = "ttnn.sum"(%ag1) <{dim_arg = [0 : i32], keep_dim = true}>
        : (tensor<4x32x512xbf16, #bf16_3d_g>) -> tensor<1x32x512xbf16, #bf16_3d>

    // CHECK: "ttnn.all_gather"
    // CHECK-SAME: bfp_bf8
    %ag2 = "ttnn.all_gather"(%s) <{
        all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}>
        : (tensor<1x32x512xbf16, #bf16_3d>) -> tensor<1x32x1024xbf16, #bf16_3d_p>

    // The pass inserts ttnn.typecast back to bf16 immediately before argmax.
    // CHECK: %[[CAST:.*]] = "ttnn.typecast"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.argmax"(%[[CAST]])
    %out = "ttnn.argmax"(%ag2) <{dim = 2 : i32, keep_dim = true, use_multicore = true}>
        : (tensor<1x32x1024xbf16, #bf16_3d_p>) -> tensor<1x32x1xui32, #u32_out>

    return %out : tensor<1x32x1xui32, #u32_out>
  }
}
