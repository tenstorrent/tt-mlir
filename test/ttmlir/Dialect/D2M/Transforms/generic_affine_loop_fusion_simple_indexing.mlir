// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-affine-loop-fusion %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> ()>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>

// Test: Two generics with matching 2D loop nests and shared intermediate memref.
// Producer stores to %intermediate, consumer loads from it.
// Both have identical indices and matching loop structure.
// CHECK-LABEL: func.func @test_basic_fusion
// CHECK: d2m.generic
// CHECK-SAME: d2m.affine_fused
// Verify producer ops (load from dram stream) precede consumer ops in fused body.
// CHECK: d2m.remote_load{{.*}}#dram
// CHECK: d2m.remote_store
// CHECK: d2m.remote_load
// CHECK: d2m.remote_store
// CHECK-NOT: d2m.generic
func.func @test_basic_fusion(
    %input: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %cb_in = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %stream_in = "d2m.stream_layout"(%input, %cb_in) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>

  // Producer: reads from input, stores to intermediate
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
      outs(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %stream_in[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %intermediate[%i, %j] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  // Consumer: reads from intermediate, stores to output
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %intermediate[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %output[%i, %j] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }

  return
}
