// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-operation-validation-and-fallback %s | FileCheck %s
module @NlpCreateQkvHeadsDecode attributes {} {
  func.func @main(
    %arg0: tensor<1x1x32x3072xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x3072xbf16, #ttnn.buffer_type<dram>>, <interleaved>>>
      {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "input"})
    -> (tensor<1x32x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.shard_status = #ttcore.shard_status<unsharded>},
       tensor<1x32x8x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.shard_status = #ttcore.shard_status<unsharded>},
       tensor<1x32x8x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.shard_status = #ttcore.shard_status<unsharded>})
    attributes {tt.function_type = "forward_device"} {
    // Input is row major which triggers TT_FATAL in backend validation.
    // Fallback should fix the input to tile and insert revert to_layout ops
    // for all 3 outputs.
    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.nlp_create_qkv_heads_decode"
    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.to_layout"
    %query, %key, %value = "ttnn.nlp_create_qkv_heads_decode"(%arg0) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32}>
      : (tensor<1x1x32x3072xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x3072xbf16, #ttnn.buffer_type<dram>>, <interleaved>>>)
      -> (tensor<1x32x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>,
          tensor<1x32x8x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>,
          tensor<1x32x8x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>)
    return %query, %key, %value
      : tensor<1x32x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>,
        tensor<1x32x8x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>,
        tensor<1x32x8x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
  }
}
