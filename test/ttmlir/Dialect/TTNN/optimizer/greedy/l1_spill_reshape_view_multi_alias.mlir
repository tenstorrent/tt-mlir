// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-greedy-l1-spill-management %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir
//
// Test: a single forked L1 producer with TWO view-eligible reshape
// consumers. With alias-group accounting all three Values (P0, V1, V2)
// share one address slot at refcount=3 and contribute one buffer to
// `currentOccupied`. With move-based aliasing (the previous design) the
// second `allocateAddressAt(V2, P0)` would assert because P0 was already
// erased from tensorAddresses by the first reshape. This test exercises
// the path that exposed the assertion.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#ttnn_layout_dram_2d = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_l1_2d = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout_l1_3d = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 4096 + d1, d2), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout_l1_4d = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2, d3), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

module attributes {ttnn.tensor_l1_usage_cap = 1.400000e-01 : f32} {
  func.func @forked_two_views_no_spill(
      %arg0: tensor<4096x512xbf16, #ttnn_layout_dram_2d>,
      %arg1: tensor<4096x512xbf16, #ttnn_layout_dram_2d>)
      -> (tensor<1x4096x512xbf16, #ttnn_layout_l1_3d>, tensor<1x1x4096x512xbf16, #ttnn_layout_l1_4d>) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    %P0 = "ttnn.add"(%arg0, %arg1) : (tensor<4096x512xbf16, #ttnn_layout_dram_2d>, tensor<4096x512xbf16, #ttnn_layout_dram_2d>) -> tensor<4096x512xbf16, #ttnn_layout_l1_2d>

    // First view: 2D → 3D, leading-1 inserted (last+second-to-last unchanged).
    %V1 = "ttnn.reshape"(%P0) <{shape = [1 : i32, 4096 : i32, 512 : i32]}> : (tensor<4096x512xbf16, #ttnn_layout_l1_2d>) -> tensor<1x4096x512xbf16, #ttnn_layout_l1_3d>

    // Second view of the SAME producer: 2D → 4D, two leading 1s.
    // Under move-based aliasing, this `allocateAddressAt(V2, P0)` asserts
    // because P0 was removed from tensorAddresses by V1's reshape.
    %V2 = "ttnn.reshape"(%P0) <{shape = [1 : i32, 1 : i32, 4096 : i32, 512 : i32]}> : (tensor<4096x512xbf16, #ttnn_layout_l1_2d>) -> tensor<1x1x4096x512xbf16, #ttnn_layout_l1_4d>

    return %V1, %V2 : tensor<1x4096x512xbf16, #ttnn_layout_l1_3d>, tensor<1x1x4096x512xbf16, #ttnn_layout_l1_4d>
  }

  // CHECK-LABEL: func.func @forked_two_views_no_spill
  // Both reshape views remain in L1 alongside P0.
  // CHECK-NOT: "ttnn.to_memory_config"{{.*}}#dram
}
