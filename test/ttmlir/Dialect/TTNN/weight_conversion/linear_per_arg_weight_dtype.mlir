// RUN: ttmlir-opt --ttnn-weight-dtype-conversion %s | FileCheck %s

// Test per-op weight dtype override on a linear op without a global target-dtype.
// For blockfloat targets the conversion is the host-pack chain
// (from_device -> to_dtype -> to_device), not a device typecast.

#dram = #ttnn.buffer_type<dram>

module attributes {} {
  // CHECK-LABEL: func.func @test_linear_per_arg_bfp4
  func.func @test_linear_per_arg_bfp4(%arg0: tensor<2048x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg1: tensor<1024x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {

    // CHECK: %[[DEV:.*]] = "ttnn.get_device"
    %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // CHECK: %[[FROM_DEV:.*]] = "ttnn.from_device"(%arg1)
    // CHECK: %[[TO_DTYPE:.*]] = "ttnn.to_dtype"(%[[FROM_DEV]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf4>
    // CHECK-SAME: -> tensor<1024x2048x!ttcore.tile<32x32, bfp_bf4>,
    // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[TO_DTYPE]], %[[DEV]])
    // CHECK-NOT: "ttnn.typecast"

    // CHECK: "ttnn.linear"(%arg0, %[[TO_DEV]], %arg2)
    // CHECK-SAME: -> tensor<2048x1024xbf16,
    %0 = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = true}> {ttcore.weight_dtype = "bfp_bf4"} : (tensor<2048x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1024x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    return %0 : tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
