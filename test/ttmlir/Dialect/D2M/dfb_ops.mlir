// RUN: ttmlir-opt %s | FileCheck %s
// Parser round-trip + verifier sanity for the D2M DFB generic-region ops:
//   d2m.get_dfb / d2m.dfb_wait / d2m.dfb_reserve / d2m.dfb_push /
//   d2m.dfb_pop / d2m.dfb_finish
// Mirrors test/ttmlir/Dialect/D2M/circular_buffer_type.mlir for the CB
// family.

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK-LABEL: @test_d2m_dfb_ops_1p1c
func.func @test_d2m_dfb_ops_1p1c(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
  ins(%arg0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
  outs(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^datamovement:
    // CHECK: d2m.get_dfb(0)
    %dfb0 = d2m.get_dfb(0) : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
    // CHECK: d2m.dfb_reserve
    %m0 = d2m.dfb_reserve %dfb0 : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK: d2m.dfb_push
    d2m.dfb_push %dfb0 : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
    // CHECK: d2m.dfb_finish
    d2m.dfb_finish %dfb0 : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
  }, {
  ^compute:
    %dfb1 = d2m.get_dfb(1) : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
    // CHECK: d2m.dfb_wait
    %m1 = d2m.dfb_wait %dfb1 : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK: d2m.dfb_pop
    d2m.dfb_pop %dfb1 : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
  }

  return
}

// CHECK-LABEL: @test_d2m_get_dfb_with_consumer_slot
func.func @test_d2m_get_dfb_with_consumer_slot(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>, #d2m.thread<compute>]}
  ins(%arg0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
  outs(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^compute_a:
    // CHECK: %[[TID:.+]] = d2m.my_thread_id
    %tid = d2m.my_thread_id : index
    // CHECK: d2m.get_dfb(0, %[[TID]])
    %dfb = d2m.get_dfb(0, %tid) {num_consumers = 4 : i64} : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
    %m = d2m.dfb_wait %dfb : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    d2m.dfb_pop %dfb : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
  }, {
  ^compute_b:
  }

  return
}

// CHECK-LABEL: @test_d2m_get_dfb_resolution_stage
func.func @test_d2m_get_dfb_resolution_stage(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
  ins(%arg0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
  outs(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^datamovement:
    // CHECK: d2m.get_dfb(0) resolution_stage = compile
    %dfb = d2m.get_dfb(0) resolution_stage = compile : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
    %m = d2m.dfb_reserve %dfb : !d2m.dfb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^compute:
  }

  return
}
