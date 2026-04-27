// RUN: ttmlir-opt --ttnn-weight-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

// Test mixed per-op and global weight dtype. Two matmuls:
// - First matmul: op has per-op annotation "bfp_bf4" → host-pack chain to bfp4
// - Second matmul: op has no annotation → falls back to global bfp8 host-pack
// Both blockfloat targets emit the from_device -> to_dtype -> to_device chain
// rather than a device typecast.

#dram = #ttnn.buffer_type<dram>

module attributes {} {
  // CHECK-LABEL: func.func @test_mixed_per_arg_and_global
  func.func @test_mixed_per_arg_and_global(
    %arg0: tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
    %arg1: tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg2: tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}
  ) -> (tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) {

    // CHECK: %[[DEV:.*]] = "ttnn.get_device"
    %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // First matmul: per-op "bfp_bf4" annotation → host-pack chain to bfp4.
    // CHECK: %[[FROM_DEV1:.*]] = "ttnn.from_device"(%arg1)
    // CHECK: %[[TO_DTYPE1:.*]] = "ttnn.to_dtype"(%[[FROM_DEV1]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf4>
    // CHECK: %[[TO_DEV1:.*]] = "ttnn.to_device"(%[[TO_DTYPE1]], %[[DEV]])

    // CHECK: %[[MM1:.*]] = "ttnn.matmul"(%arg0, %[[TO_DEV1]])
    %0 = "ttnn.matmul"(%arg0, %arg1) {ttcore.weight_dtype = "bfp_bf4"} : (tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    // Second matmul: no per-op annotation, falls back to global bfp8.
    // CHECK: %[[FROM_DEV2:.*]] = "ttnn.from_device"(%arg2)
    // CHECK: %[[TO_DTYPE2:.*]] = "ttnn.to_dtype"(%[[FROM_DEV2]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    // CHECK: %[[TO_DEV2:.*]] = "ttnn.to_device"(%[[TO_DTYPE2]], %[[DEV]])
    // CHECK-NOT: "ttnn.typecast"

    // CHECK: %[[MM2:.*]] = "ttnn.matmul"(%arg0, %[[TO_DEV2]])
    %1 = "ttnn.matmul"(%arg0, %arg2) : (tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    return %0, %1 : tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
