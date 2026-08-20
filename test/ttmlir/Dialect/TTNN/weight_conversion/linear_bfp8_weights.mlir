// RUN: ttmlir-opt --ttnn-weight-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

// Test that the BFP8 weight conversion pass correctly:
// 1. Inserts a single on-device typecast before the linear for bfp8 targets.
// 2. Converts the weight tensor (B operand) to bfp_bf8
// 3. Updates the linear to use the resulting on-device tensor
// 4. Keeps the output of linear as bf16 (unchanged)
// 5. Keeps the activation input (A operand) and bias of linear as bf16 (unchanged)

#dram = #ttnn.buffer_type<dram>

module attributes {} {
  func.func @test_linear_bfp8_weights(%arg0: tensor<1024xbf16, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg1: tensor<1024x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<2048x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg3: tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {

    // CHECK-LABEL: func.func @test_linear_bfp8_weights

    %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // CHECK-NOT: "ttnn.from_device"
    // CHECK: %[[TYPECAST:.*]] = "ttnn.typecast"(%arg1)
    // CHECK-SAME: -> tensor<1024x2048x!ttcore.tile<32x32, bfp_bf8>,
    // CHECK-NOT: "ttnn.to_device"

    // CHECK: "ttnn.linear"(%arg2, %[[TYPECAST]], %arg3)
    // CHECK-SAME: -> tensor<2048x1024xbf16,
    %0 = "ttnn.linear"(%arg2, %arg1, %arg3) <{transpose_a = false, transpose_b = true}> : (tensor<2048x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1024x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    // CHECK: return {{.*}} : tensor<2048x1024xbf16,
    return %0 : tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
