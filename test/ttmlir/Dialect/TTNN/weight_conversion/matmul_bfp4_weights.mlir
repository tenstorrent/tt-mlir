// RUN: ttmlir-opt --ttnn-weight-dtype-conversion="target-dtype=bfp_bf4" %s | FileCheck %s

// Test that the weight dtype conversion pass correctly:
// 1. Inserts a ttnn.typecast operation before the matmul
// 2. Converts the weight tensor (B operand) to bfp_bf4
// 3. Updates the matmul to use the typecast result
// 4. Keeps the output of matmul as bf16 (unchanged)
// 5. Keeps the activation input (A operand) of matmul as bf16 (unchanged)

#dram = #ttnn.buffer_type<dram>

module attributes {} {
  func.func @test_matmul_bfp4_weights(%arg0: tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg1: tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {

    // CHECK-LABEL: func.func @test_matmul_bfp4_weights

    // CHECK: %[[TYPECAST:.*]] = "ttnn.typecast"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf4>
    // CHECK-SAME: -> tensor<1x128x256x!ttcore.tile<32x32, bfp_bf4>,

    // CHECK: "ttnn.matmul"(%arg0, %[[TYPECAST]])
    // CHECK-SAME: -> tensor<1x32x256xbf16,
    %0 = "ttnn.matmul"(%arg0, %arg1) : (tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    // CHECK: return {{.*}} : tensor<1x32x256xbf16,
    return %0 : tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
