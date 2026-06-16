// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>

#cache    = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#permuted = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#repeated = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32 + d2, d3), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#concat   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 64 + d2, d3), <1x1>, memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @test_read_path_bfp8
// CHECK-SAME: %arg0: tensor<1x1x64x32x!ttcore.tile<32x32, bfp_bf8>,
// CHECK-SAME: %arg1: tensor<1x2x32x64xbf16,
module attributes {} {
  func.func @test_read_path_bfp8(
      %arg0: tensor<1x1x64x32xbf16, #cache> {ttcore.kv_cache},
      %arg1: tensor<1x2x32x64xbf16, #repeated>
  ) attributes {tt.function_type = "forward_device"} {

    // CHECK: %[[PERMUTE:.*]] = "ttnn.permute"(%arg0)
    // CHECK-SAME: -> tensor<1x1x32x64x!ttcore.tile<32x32, bfp_bf8>,
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}> : (
        tensor<1x1x64x32xbf16, #cache>
    ) -> tensor<1x1x32x64xbf16, #permuted>

    // CHECK: %[[REPEAT:.*]] = "ttnn.repeat"(%[[PERMUTE]])
    // CHECK-SAME: -> tensor<1x2x32x64x!ttcore.tile<32x32, bfp_bf8>,
    %1 = "ttnn.repeat"(%0) <{repeat_dims = #ttnn.shape<1x2x1x1>}> : (
        tensor<1x1x32x64xbf16, #permuted>
    ) -> tensor<1x2x32x64xbf16, #repeated>

    // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    // CHECK-SAME: -> tensor<1x2x32x64x!ttcore.tile<32x32, bfp_bf8>,
    // CHECK: "ttnn.concat"(%[[CAST]], %[[REPEAT]])
    // CHECK-SAME: -> tensor<1x2x64x64x!ttcore.tile<32x32, bfp_bf8>,
    %2 = "ttnn.concat"(%arg1, %1) <{dim = 2 : si32}> : (
        tensor<1x2x32x64xbf16, #repeated>,
        tensor<1x2x32x64xbf16, #repeated>
    ) -> tensor<1x2x64x64xbf16, #concat>

    return
  }
}
