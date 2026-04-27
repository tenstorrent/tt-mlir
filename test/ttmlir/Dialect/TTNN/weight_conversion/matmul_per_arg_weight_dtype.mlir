// RUN: ttmlir-opt --ttnn-weight-dtype-conversion %s | FileCheck %s

// Test per-op weight dtype override without a global target-dtype.
// The "ttcore.weight_dtype" discardable attribute on the matmul op should drive
// the conversion. For blockfloat targets the conversion is the host-pack
// chain (from_device -> to_dtype -> to_device), not a device typecast.

#dram = #ttnn.buffer_type<dram>

module attributes {} {
  // CHECK-LABEL: func.func @test_per_arg_bfp8
  func.func @test_per_arg_bfp8(%arg0: tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg1: tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {

    // CHECK: %[[DEV:.*]] = "ttnn.get_device"
    %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // CHECK: %[[FROM_DEV:.*]] = "ttnn.from_device"(%arg1)
    // CHECK: %[[TO_DTYPE:.*]] = "ttnn.to_dtype"(%[[FROM_DEV]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    // CHECK-SAME: -> tensor<1x128x256x!ttcore.tile<32x32, bfp_bf8>,
    // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[TO_DTYPE]], %[[DEV]])
    // CHECK-NOT: "ttnn.typecast"

    // CHECK: "ttnn.matmul"(%arg0, %[[TO_DEV]])
    // CHECK-SAME: -> tensor<1x32x256xbf16,
    %0 = "ttnn.matmul"(%arg0, %arg1) {ttcore.weight_dtype = "bfp_bf8"} : (tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    return %0 : tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
