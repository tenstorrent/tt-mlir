// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test that d2m-allocate correctly handles d2m.generic operations with empty
// indexing_maps, i.e., stream is inserted.
// The pass should skip reduction/broadcast dimension analysis
// but still insert streams based on other factors (e.g., memory space).

#l1_ = #ttcore.memory_space<l1>
#dram_ = #ttcore.memory_space<dram>

module {
  // CHECK-LABEL: func.func @empty_indexing_maps_dram_input
  func.func @empty_indexing_maps_dram_input(
    %in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #dram_>,
    %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>
  ) {
    // CHECK: %[[ALLOC:.*]] = memref.alloc()
    // CHECK: %[[STREAM:.*]] = "d2m.stream_layout"(%{{.*}}, %[[ALLOC]]) : {{.*}}#dram>
    // d2m.generic with empty indexing_maps
    // CHECK: d2m.generic
    // CHECK-SAME: indexing_maps = []
    // CHECK: ins(%[[STREAM]] : {{.*}})
    // CHECK: outs(%{{.*}} : {{.*}})
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [],
      iterator_types = [],
      threads = [#d2m.thread<compute>]
    }
    ins(%in0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #dram_>)
    outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #dram_>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %val = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #dram_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #dram_>
      %out = d2m.reserve %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}
