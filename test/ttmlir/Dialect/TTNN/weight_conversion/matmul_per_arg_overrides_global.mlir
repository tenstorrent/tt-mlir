// RUN: ttmlir-opt --ttnn-weight-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

// Test that per-op "ttcore.weight_dtype" attribute overrides the global
// target-dtype. Here the global is bfp_bf8, but the op annotation says bfp_bf4.

#dram = #ttnn.buffer_type<dram>

module attributes {} {
  // CHECK-LABEL: func.func @test_per_arg_overrides_global
  func.func @test_per_arg_overrides_global(%arg0: tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg1: tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {

    // CHECK: %[[DEV:.*]] = "ttnn.get_device"
    %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // Per-op says bfp4, global says bfp8. Per-op should win and emit the
    // host-side chain to bfp4.
    // CHECK: %[[FROM_DEV:.*]] = "ttnn.from_device"(%arg1)
    // CHECK: %[[TYPECAST:.*]] = "ttnn.typecast"(%[[FROM_DEV]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf4>
    // CHECK-SAME: -> tensor<1x128x256x!ttcore.tile<32x32, bfp_bf4>,
    // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[TYPECAST]], %[[DEV]])
    // CHECK-NOT: supportedDataTypes<bfp_bf8>

    // CHECK: "ttnn.matmul"(%arg0, %[[TO_DEV]])
    // CHECK-SAME: -> tensor<1x32x256xbf16,
    %0 = "ttnn.matmul"(%arg0, %arg1) {ttcore.weight_dtype = "bfp_bf4"} : (tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    return %0 : tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
