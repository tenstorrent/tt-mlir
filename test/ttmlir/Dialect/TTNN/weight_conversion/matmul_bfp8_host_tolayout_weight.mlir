// RUN: ttmlir-opt --ttnn-weight-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

// Verify that blockfloat weight conversion packs directly from a host
// ToLayoutOp input when available, avoiding a to_device -> from_device round
// trip for CPU-hoisted const-eval weights before layout decomposition.

#dram = #ttnn.buffer_type<dram>
#system = #ttnn.buffer_type<system_memory>

module attributes {} {
  func.func @test_matmul_bfp8_host_tolayout_weight(
      %arg0: tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
      %arg1: tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x128x256xbf16, #system>>> {ttcore.argument_type = #ttcore.argument_type<parameter>})
      -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {

    // CHECK: #[[HOST_TILED:.*]] = #ttnn.ttnn_layout<{{.*}}!ttcore.tile<32x32, bf16>, #system_memory
    // CHECK-LABEL: func.func @test_matmul_bfp8_host_tolayout_weight
    // CHECK: %[[DEV:.*]] = "ttnn.get_device"
    %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // CHECK: %[[HOST_LAYOUT:.*]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: -> tensor<1x128x256xbf16, #[[HOST_TILED]]>
    // CHECK-NOT: "ttnn.from_device"
    // CHECK: %[[TYPECAST:.*]] = "ttnn.typecast"(%[[HOST_LAYOUT]])
    // CHECK-SAME: -> tensor<1x128x256x!ttcore.tile<32x32, bfp_bf8>,
    // CHECK: %[[TO_DEV:.*]] = "ttnn.to_device"(%[[TYPECAST]], %[[DEV]])
    // CHECK: "ttnn.matmul"(%arg0, %[[TO_DEV]])
    // CHECK-SAME: -> tensor<1x32x256xbf16,
    %weight = "ttnn.to_layout"(%arg1) : (tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x128x256xbf16, #system>>>) -> tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
    %0 = "ttnn.matmul"(%arg0, %weight) : (tensor<1x32x128xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x128x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    return %0 : tensor<1x32x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
