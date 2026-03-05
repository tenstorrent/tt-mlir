// Negative test ensuring that back-to-back d2m.view_layout ops in an operand
// defining chain are rejected by the allocator under the "infer" stream-insert
// policy.
// RUN: not ttmlir-opt --ttcore-register-device "--d2m-allocate=stream-insert-policy=infer" %s 2>&1 | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#remap4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>

module {
  // CHECK: error: 'd2m.generic' op operand #0 has a defining chain with back-to-back d2m.view_layout ops
  func.func @test_back_to_back_view_layout_chain(%arg0: memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>) {
    %view0 = d2m.view_layout %arg0 remapping = #remap4 : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1> -> memref<1x1x32x32xf32, #ttcore.view<4>, #l1>
    %view1 = d2m.view_layout %view0 remapping = #remap4 : memref<1x1x32x32xf32, #ttcore.view<4>, #l1> -> memref<1x1x32x32xf32, #ttcore.view<4>, #l1>
    %out = memref.alloc() : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>]}
        ins(%view1 : memref<1x1x32x32xf32, #ttcore.view<4>, #l1>)
        outs(%out : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>) {
    ^datamovement0(%cb0: !d2m.cb<memref<32x32xf32, #l1>>, %cb1: !d2m.cb<memref<32x32xf32, #l1>>):
      %buf = d2m.reserve %cb1 : !d2m.cb<memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
    }
    return
  }
}
