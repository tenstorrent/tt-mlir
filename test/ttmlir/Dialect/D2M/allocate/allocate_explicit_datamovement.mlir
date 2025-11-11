// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test that d2m-allocate correctly handles d2m.generic operations in explicit
// datamovement form (empty block_factors, indexing_maps, and iterator_types).
// The pass should skip reduction/broadcast dimension analysis for such ops
// but accept existing operand streams.

#l1_ = #ttcore.memory_space<l1>
#dram_ = #ttcore.memory_space<dram>


module {
  // CHECK-LABEL: func.func @empty_indexing_maps_dram_input
  func.func @empty_indexing_maps_dram_input(
    %arg_in: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #dram_>,
    %arg_out: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
  ) {
    // CHECK: %[[ALLOC:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #l1>
    %in_buf = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
    // CHECK: %[[STREAM:.*]] = "d2m.stream_layout"(%{{.*}}, %[[ALLOC]]) : {{.*}}#dram>
    %in_stream = "d2m.stream_layout"(%arg_in, %in_buf) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #dram_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram_>
    // d2m.generic with empty indexing_maps is valid IR:
    // CHECK: d2m.generic
    // CHECK-SAME: indexing_maps = []
    // CHECK: ins(%[[STREAM]] : {{.*}})
    // CHECK: outs(%{{.*}} : {{.*}})
    d2m.generic {
      block_factors = [],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [],
      iterator_types = [],
      threads = [#d2m.thread<compute>]
    }
    ins(%in_stream : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram_>)
    outs(%arg_out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #dram_>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %val = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #dram_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #dram_>
      %out = d2m.reserve %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}
