// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that an alloc inside a compute loop (`scf.for` / `affine.for`
// without `d2m.blocking_loop`) is recognized as the output operand alloc
// when it backs the localBuffer of a `d2m.remote_store` for the output.
//
// Before the fix, d2m.GenericOp::getOperandAlloc refused to descend into
// compute loops, so the alloc was never stamped with CBLayoutAttr/address.
// Downstream HoistCBAllocs then left it inside the generic and
// SplitUnifiedThread crashed.

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  // The output alloc lives inside a compute `scf.for` (no `d2m.blocking_loop`)
  // and is the localBuffer of the output's `d2m.remote_store`. Allocate
  // must stamp it with an `address`.
  // CHECK-LABEL: func.func @scf_output_alloc_in_compute_loop
  func.func @scf_output_alloc_in_compute_loop(
      %in: memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>,
      %out: memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %out_stream = d2m.view_layout %out remapping = #map4 : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    // CHECK: d2m.generic
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
        outs(%out_stream : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    ^unified0():
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      // CHECK: scf.for
      // CHECK-NOT: d2m.blocking_loop
      scf.for %iv = %c0 to %c1 step %c1 {
        // CHECK: memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64, d2m.synchronized_buffer = 2 : i64} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
        %alloc = memref.alloc() {alignment = 16 : i64, d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
        // CHECK: d2m.remote_store
        d2m.remote_store %out_stream[%core0, %core1] %alloc : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>
      }
    }
    return
  }

  // Same shape, with `affine.for` instead of `scf.for`.
  // CHECK-LABEL: func.func @affine_output_alloc_in_compute_loop
  func.func @affine_output_alloc_in_compute_loop(
      %in: memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>,
      %out: memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %out_stream = d2m.view_layout %out remapping = #map4 : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    // CHECK: d2m.generic
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
        outs(%out_stream : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    ^unified0():
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      // CHECK: affine.for
      // CHECK-NOT: d2m.blocking_loop
      affine.for %iv = 0 to 1 {
        // CHECK: memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64, d2m.synchronized_buffer = 2 : i64} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
        %alloc = memref.alloc() {alignment = 16 : i64, d2m.synchronized_buffer = 2} : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
        // CHECK: d2m.remote_store
        d2m.remote_store %out_stream[%core0, %core1] %alloc : memref<2x2x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>
      }
    }
    return
  }
}
