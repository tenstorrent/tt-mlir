// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-affine-loop-fusion="enable=true" %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> ()>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_add = affine_map<(d0)[s0] -> (d0 + s0)>

// Test: Two generics with matching 2D loop nests and shared intermediate memref.
// Producer stores to %intermediate, consumer loads from it.
// Both have identical indices and matching loop structure.
// CHECK-LABEL: func.func @test_basic_fusion
// CHECK: d2m.generic
// CHECK: d2m.block_offset
// Verify producer ops (load from dram stream) precede consumer ops in fused body.
// CHECK: d2m.remote_load{{.*}}#dram
// CHECK: d2m.remote_store
// CHECK: d2m.remote_load
// CHECK: d2m.remote_store
// CHECK-NOT: d2m.block_offset_dim
// CHECK-NOT: d2m.generic
func.func @test_basic_fusion(
    %input: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %cb_in = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %stream_in = "d2m.stream_layout"(%input, %cb_in) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>

  // Producer: reads from input, applies relu, stores to intermediate
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
      outs(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %off0 = d2m.block_offset(0) : index
        %idx0 = affine.apply #map_add(%i)[%off0]
        %off1 = d2m.block_offset(1) : index
        %idx1 = affine.apply #map_add(%j)[%off1]
        %loaded = d2m.remote_load %buf_in %stream_in[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%loaded : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%buf_out : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
        {
        ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %relu = "d2m.tile_relu"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %relu : !ttcore.tile<32x32, f32>
        }

        %off0_store = d2m.block_offset(0) : index
        %idx0_store = affine.apply #map_add(%i)[%off0_store]
        %off1_store = d2m.block_offset(1) : index
        %idx1_store = affine.apply #map_add(%j)[%off1_store]
        %stored = d2m.remote_store %intermediate[%idx0_store, %idx1_store] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  // Consumer: reads from intermediate, applies exp, stores to output
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %off0 = d2m.block_offset(0) : index
        %idx0 = affine.apply #map_add(%i)[%off0]
        %off1 = d2m.block_offset(1) : index
        %idx1 = affine.apply #map_add(%j)[%off1]
        %loaded = d2m.remote_load %buf_in %intermediate[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%loaded : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%buf_out : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
        {
        ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %exp = "d2m.tile_exp"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %exp : !ttcore.tile<32x32, f32>
        }

        %off0_store = d2m.block_offset(0) : index
        %idx0_store = affine.apply #map_add(%i)[%off0_store]
        %off1_store = d2m.block_offset(1) : index
        %idx1_store = affine.apply #map_add(%j)[%off1_store]
        %stored = d2m.remote_store %output[%idx0_store, %idx1_store] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  return
}

// Test: Multiple readers of intermediate should prevent fusion.
// CHECK-LABEL: func.func @test_no_fusion_multiple_readers
// CHECK: d2m.generic
// CHECK: d2m.generic
// CHECK: d2m.generic
func.func @test_no_fusion_multiple_readers(
    %input: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output1 = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output2 = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %cb_in = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %stream_in = "d2m.stream_layout"(%input, %cb_in) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>

  // Producer
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
      outs(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %off0 = d2m.block_offset(0) : index
        %off1 = d2m.block_offset(1) : index
        %idx0 = affine.apply #map_add(%i)[%off0]
        %idx1 = affine.apply #map_add(%j)[%off1]
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %stream_in[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %intermediate[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  // Consumer 1: reads from intermediate
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %off0 = d2m.block_offset(0) : index
        %off1 = d2m.block_offset(1) : index
        %idx0 = affine.apply #map_add(%i)[%off0]
        %idx1 = affine.apply #map_add(%j)[%off1]
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %intermediate[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %output1[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  // Consumer 2: also reads from intermediate — prevents fusion
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb4: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb5: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %off0 = d2m.block_offset(0) : index
        %off1 = d2m.block_offset(1) : index
        %idx0 = affine.apply #map_add(%i)[%off0]
        %idx1 = affine.apply #map_add(%j)[%off1]
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %intermediate[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %output2[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  return
}

// Test: Reader hidden behind two view_layout ops should prevent fusion.
// CHECK-LABEL: func.func @test_no_fusion_chained_view_reader
// CHECK: d2m.generic
// CHECK: d2m.generic
// CHECK: d2m.generic
func.func @test_no_fusion_chained_view_reader(
    %input: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output1 = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output2 = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %cb_in = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %stream_in = "d2m.stream_layout"(%input, %cb_in) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>

  // Producer
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
      outs(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %off0 = d2m.block_offset(0) : index
        %off1 = d2m.block_offset(1) : index
        %idx0 = affine.apply #map_add(%i)[%off0]
        %idx1 = affine.apply #map_add(%j)[%off1]
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %stream_in[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %intermediate[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  %view1 = d2m.view_layout %intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  %view2 = d2m.view_layout %view1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>

  // Consumer 1: direct reader from intermediate.
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified1(%cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %off0 = d2m.block_offset(0) : index
        %off1 = d2m.block_offset(1) : index
        %idx0 = affine.apply #map_add(%i)[%off0]
        %idx1 = affine.apply #map_add(%j)[%off1]
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %intermediate[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %output1[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  // Consumer 2: hidden reader through a two-level view chain.
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%view2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>)
      outs(%output2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified2(%cb4: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb5: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %off0 = d2m.block_offset(0) : index
        %off1 = d2m.block_offset(1) : index
        %idx0 = affine.apply #map_add(%i)[%off0]
        %idx1 = affine.apply #map_add(%j)[%off1]
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %view2[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %output2[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  return
}

// Test: Fuse matmul (MNK, 3 blocking loops) producer into binary add (MN, 2 blocking loops) consumer.
// Matmul is superset (3 loops), add is subset (2 loops).
// Fused generic should use matmul's block_factors [1, 1, 2].
// CHECK-LABEL: func.func @test_matmul_add_subset_fusion
// CHECK: d2m.generic
// CHECK-SAME: block_factors = [1, 1, 2]
// CHECK-NOT: d2m.generic
func.func @test_matmul_add_subset_fusion(
    %lhs: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>,
    %rhs: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>,
    %bias: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %output = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %cb_lhs = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %cb_rhs = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %stream_lhs = "d2m.stream_layout"(%lhs, %cb_lhs) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>
  %stream_rhs = "d2m.stream_layout"(%rhs, %cb_rhs) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>

  // Producer: matmul with 3 blocking loops (M, N, K)
  d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
      ins(%stream_lhs, %stream_rhs : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>)
      outs(%intermediate : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    %bf2 = d2m.get_block_factor(2) : index
    affine.for %m = 0 to %bf0 {
      affine.for %n = 0 to %bf1 {
        %buf_out = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

        affine.for %k = 0 to %bf2 {
          %buf_lhs = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %buf_rhs = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %off0 = d2m.block_offset(0) : index
          %idx0 = affine.apply #map_add(%m)[%off0]
          %off2 = d2m.block_offset(2) : index
          %idx2 = affine.apply #map_add(%k)[%off2]
          %loaded_lhs = d2m.remote_load %buf_lhs %stream_lhs[%idx0, %idx2]      : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %off1 = d2m.block_offset(1) : index
          %idx1 = affine.apply #map_add(%n)[%off1]
          %off2_rhs = d2m.block_offset(2) : index
          %idx2_rhs = affine.apply #map_add(%k)[%off2_rhs]
          %loaded_rhs = d2m.remote_load %buf_rhs %stream_rhs[%idx2_rhs, %idx1]      : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          "d2m.tile_matmul_block"(%loaded_lhs, %loaded_rhs, %buf_out) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
        } {d2m.blocking_loop = 2}

        %off0_store = d2m.block_offset(0) : index
        %idx0_store = affine.apply #map_add(%m)[%off0_store]
        %off1_store = d2m.block_offset(1) : index
        %idx1_store = affine.apply #map_add(%n)[%off1_store]
        %stored = d2m.remote_store %intermediate[%idx0_store, %idx1_store] %buf_out : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>

      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  // Consumer: binary add with 2 blocking loops (M, N)
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate, %bias : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>)
      outs(%output : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
  ^unified1(%cb3: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb4: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb5: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    %abf0 = d2m.get_block_factor(0) : index
    %abf1 = d2m.get_block_factor(1) : index
    affine.for %am = 0 to %abf0 {
      affine.for %an = 0 to %abf1 {
        %buf_inter = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        %aoff0 = d2m.block_offset(0) : index
        %aidx0 = affine.apply #map_add(%am)[%aoff0]
        %aoff1 = d2m.block_offset(1) : index
        %aidx1 = affine.apply #map_add(%an)[%aoff1]
        %loaded_inter = d2m.remote_load %buf_inter %intermediate[%aidx0, %aidx1] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

        %buf_bias = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        %aoff0_bias = d2m.block_offset(0) : index
        %aidx0_bias = affine.apply #map_add(%am)[%aoff0_bias]
        %aoff1_bias = d2m.block_offset(1) : index
        %aidx1_bias = affine.apply #map_add(%an)[%aoff1_bias]
        %loaded_bias = d2m.remote_load %buf_bias %bias[%aidx0_bias, %aidx1_bias] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

        %buf_add_out = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

        linalg.generic {
          indexing_maps = [#map, #map, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%buf_bias, %buf_inter : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%buf_add_out : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
        {
        ^bb0(%in1_val: !ttcore.tile<32x32, f32>, %in2_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %add = "d2m.tile_add"(%in1_val, %in2_val) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %add : !ttcore.tile<32x32, f32>
        }

        %aoff0_store = d2m.block_offset(0) : index
        %aidx0_store = affine.apply #map_add(%am)[%aoff0_store]
        %aoff1_store = d2m.block_offset(1) : index
        %aidx1_store = affine.apply #map_add(%an)[%aoff1_store]
        %stored_out = d2m.remote_store %output[%aidx0_store, %aidx1_store] %buf_add_out : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  return
}

// Test: producer/consumer index mismatch on the shared memref must not fuse.
// CHECK-LABEL: func.func @test_store_load_index_mismatch_ab
// CHECK: d2m.generic
// CHECK: d2m.generic
func.func @test_store_load_index_mismatch_ab(
    %input: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %cb_in = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %stream_in = "d2m.stream_layout"(%input, %cb_in) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>

  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
      outs(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %off0 = d2m.block_offset(0) : index
        %off1 = d2m.block_offset(1) : index
        %idx0 = affine.apply #map_add(%i)[%off0]
        %idx1 = affine.apply #map_add(%j)[%off1]
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %stream_in[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%loaded : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%buf_out : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
        {
        ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %relu = "d2m.tile_relu"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %relu : !ttcore.tile<32x32, f32>
        }
        %stored = d2m.remote_store %intermediate[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified1(%cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %off0 = d2m.block_offset(0) : index
        %off1 = d2m.block_offset(1) : index
        %idx0 = affine.apply #map_add(%i)[%off0]
        %idx1 = affine.apply #map_add(%j)[%off1]
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %intermediate[%idx1, %idx0] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%loaded : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%buf_out : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
        {
        ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %exp = "d2m.tile_exp"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %exp : !ttcore.tile<32x32, f32>
        }
        %stored = d2m.remote_store %output[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  return
}
