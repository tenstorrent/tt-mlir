// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access-unscheduled --d2m-insert-dst-register-access-scheduled %s | FileCheck %s
// (The no-`linalg_root` fallback is owned by the scheduled pass, so the scheduled invocation does the work; the unscheduled one is a no-op here but is included for parity with the production pipeline order.)

// Test the fallback path where there's no d2m.linalg_root loop
// but compute ops exist. The pass should still insert
// acquire_dst and move it to just before its first use.

#l1_ = #ttcore.memory_space<l1>
#shard = #ttcore.shard<4096x4096, 1>

module {
  // CHECK-LABEL: func.func @arange_no_loop
  func.func @arange_no_loop(
      %in: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard, #l1_>,
      %out: memref<1x4x1x1x!ttcore.tile<32x32, f32>, #shard, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard, #l1_>)
        outs(%out : memref<1x4x1x1x!ttcore.tile<32x32, f32>, #shard, #l1_>)
     {
      %c4096 = arith.constant 4096 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %alloc_cb = memref.alloc() {address = 103712 : i64, alignment = 16 : i64, d2m.synchronized_buffer = 2 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      // Verify ordering: remote_load must come BEFORE acquire_dst
      // CHECK: d2m.remote_load
      d2m.remote_load %alloc_cb %in[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard, #l1_>
      %alloc_out = d2m.operand_alias %out : memref<1x4x1x1x!ttcore.tile<32x32, f32>, #shard, #l1_> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      // CHECK: d2m.fill_arange_tile
      d2m.fill_arange_tile to %alloc_cb : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index

      // Load tile from CB - this value will be moved to DST for the compute op
      // CHECK: memref.load
      %tile = memref.load %alloc_cb[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

      // Compute offset for arange
      %1 = arith.muli %core0, %c4096 : index
      %2 = arith.muli %core1, %c32 : index
      %3 = arith.addi %1, %2 : index
      %4 = arith.index_cast %3 : index to i64
      %5 = arith.sitofp %4 : i64 to f32

      // acquire_dst should be moved to just before its first use (after the ops above)
      // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #{{.*}}>
      // CHECK: affine.store %{{.*}}, %[[DST]][0]
      // CHECK: %[[LOADED:.*]] = affine.load %[[DST]][0]
      // CHECK: %[[RESULT:.*]] = "d2m.tile_add"(%[[LOADED]], %{{.*}})
      // CHECK: affine.store %[[RESULT]], %[[DST]][0]
      // CHECK: %[[FINAL:.*]] = affine.load %[[DST]][0]
      // CHECK: memref.store %[[FINAL]]
      %result = "d2m.tile_add"(%tile, %5) : (!ttcore.tile<32x32, f32>, f32) -> !ttcore.tile<32x32, f32>

      memref.store %result, %alloc_out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %core0_store = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1_store = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      d2m.remote_store %out[%core0_store, %core1_store] %alloc_out : memref<1x4x1x1x!ttcore.tile<32x32, f32>, #shard, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}
