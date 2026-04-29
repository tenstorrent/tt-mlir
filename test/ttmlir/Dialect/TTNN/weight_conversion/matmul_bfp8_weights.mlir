// RUN: ttmlir-opt --ttnn-weight-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

// Test that the BFP8 weight conversion pass correctly:
// 1. Inserts a host-side chain (from_device -> typecast -> to_device) before
//    the matmul for blockfloat targets.
// 2. Converts the weight tensor (B operand) to bfp_bf8
// 3. Updates the matmul to use the resulting on-device tensor
// 4. Keeps the output of matmul as bf16 (unchanged)
// 5. Keeps the activation input (A operand) of matmul as bf16 (unchanged)

#dram = #ttnn.buffer_type<dram>

module attributes {} {
  func.func @test_matmul_bfp8_weights(%arg0: tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg1: tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {

    // CHECK-LABEL: func.func @test_matmul_bfp8_weights

    // CHECK: %[[DEV:.*]] = "ttnn.get_device"
    %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // CHECK: %[[FROM_DEV:.*]] = "ttnn.from_device"(%arg1)
    // CHECK: %[[TYPECAST:.*]] = "ttnn.typecast"(%[[FROM_DEV]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    // CHECK-SAME: -> tensor<1x128x256x!ttcore.tile<32x32, bfp_bf8>,
    // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[TYPECAST]], %[[DEV]])

    // CHECK: "ttnn.matmul"(%arg0, %[[TO_DEV]])
    // CHECK-SAME: -> tensor<1x32x256xbf16,
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    // CHECK: return {{.*}} : tensor<1x32x256xbf16,
    return %0 : tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
