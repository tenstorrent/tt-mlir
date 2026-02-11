// RUN: ttmlir-opt --allow-unregistered-dialect --d2m-generic-affine-scalrep %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>

// Test 1: Basic store-to-load forwarding with remote_store -> remote_load
// on the same locally-allocated memref with identical constant indices.
// Both the load and store should be eliminated (along with the device memref
// alloc) since after forwarding, the only remaining users of the device
// memref are stores.
//
// CHECK-LABEL: func.func @test_basic_store_to_load_forwarding
// CHECK-NOT: d2m.remote_store
// CHECK-NOT: d2m.remote_load
// CHECK: return
func.func @test_basic_store_to_load_forwarding(
    %local_buf_a: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %device_memref = memref.alloc() : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  // Store local_buf_a to device_memref[0, 1]
  %store_result = d2m.remote_store %device_memref[%c0, %c1] %local_buf_a
    : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
      memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  // Load from device_memref[0, 1] into a fresh buffer — should be forwarded
  %local_buf_b = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
  %load_result = d2m.remote_load %local_buf_b %device_memref[%c0, %c1]
    : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
      memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

  return %load_result : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
}

// Test 2: Store-to-load forwarding inside an affine.for loop with a
// locally-allocated device memref. Both store and load should be eliminated.
//
// CHECK-LABEL: func.func @test_store_to_load_in_affine_loop
// CHECK-NOT: d2m.remote_store
// CHECK-NOT: d2m.remote_load
// CHECK: affine.for
// CHECK:   affine.for
// CHECK:     "test.use"
func.func @test_store_to_load_in_affine_loop(
    %local_buf_a: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
  %device_memref = memref.alloc() : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 4 {
      %store_result = d2m.remote_store %device_memref[%i, %j] %local_buf_a
        : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
          memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

      %local_buf_b = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %load_result = d2m.remote_load %local_buf_b %device_memref[%i, %j]
        : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
          memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
        -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

      "test.use"(%load_result) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    }
  }
  return
}

// Test 3: Different indices should NOT be forwarded.
//
// CHECK-LABEL: func.func @test_no_forwarding_different_indices
// CHECK: d2m.remote_store
// CHECK: d2m.remote_load
func.func @test_no_forwarding_different_indices(
    %local_buf_a: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %device_memref = memref.alloc() : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  %store_result = d2m.remote_store %device_memref[%c0, %c1] %local_buf_a
    : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
      memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  // Load from [1, 0] — different indices, should not forward
  %local_buf_b = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
  %load_result = d2m.remote_load %local_buf_b %device_memref[%c1, %c0]
    : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
      memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

  "test.use"(%load_result) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  return
}

// Test 4: Scalar replacement on fused generics with block_factor and block_index.
// The block_factor values are loop-invariant and can be scalar replaced.
// After conversion to compatibility form, scalar replacement, and conversion back,
// the generic op should still be present with the block_index operations.
// The test.use op mocks computation between load and store.
//
// CHECK-LABEL: func.func @test_scalar_replacement_fused_generic
// CHECK: d2m.generic
// CHECK-SAME: d2m.affine_fused
// CHECK: d2m.get_block_factor
// CHECK: affine.for
// CHECK: affine.for
// CHECK: d2m.block_index
// CHECK: d2m.remote_load
// CHECK: test.use
// CHECK: d2m.remote_store
func.func @test_scalar_replacement_fused_generic(
    %input: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  %output = memref.alloc() : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  d2m.generic {block_factors = [1, 1], d2m.affine_fused, grid = #ttcore.grid<2x4>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
      ins(%input : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %idx0 = d2m.block_index(0) : index
        %idx1 = d2m.block_index(1) : index
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %input[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        "test.use"(%loaded) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> ()
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored = d2m.remote_store %output[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }
  return
}

// CHECK-LABEL: func.func @test_multiple_block_index_same_dimension
// CHECK: d2m.generic
// CHECK: d2m.get_block_factor
// CHECK: affine.for
// CHECK: affine.for
// CHECK: d2m.block_index
// CHECK: d2m.remote_load
// CHECK: test.use
// CHECK: d2m.remote_store
func.func @test_multiple_block_index_same_dimension(
    %input: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
    %temp : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    ) {
  %output = memref.alloc() : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  d2m.generic {block_factors = [1, 1], d2m.affine_fused, grid = #ttcore.grid<2x4>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
      ins(
        %input, %temp : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> , memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index

    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        // Multiple calls to block_index(0) — should resolve to same value
        %idx0_1 = d2m.block_index(0) : index
        //%idx0_2 = d2m.block_index(0) : index
        %idx1 = d2m.block_index(1) : index

        // First load-compute-store sequence using idx0_1
        %buf_in1 = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded1 = d2m.remote_load %buf_in1 %input[%idx0_1, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        "test.use"(%loaded1) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> ()
        %buf_out1 = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored1 = d2m.remote_store %temp[%idx0_1, %idx1] %buf_out1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

        // Second load-compute-store sequence using idx0_2 (same dimension as idx0_1)
        %buf_in2 = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded2 = d2m.remote_load %buf_in2 %temp[%idx0_1, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        "test.use"(%loaded2) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> ()
        %buf_out2 = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored2 = d2m.remote_store %output[%idx0_1, %idx1] %buf_out2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }
  return
}

// Test 6: Intermediate operand internalization in a fused generic.
// %intermediate is a memref.alloc only used by this generic op (no other
// top-level users). It should be removed from the generic's inputs, its
// CB block arg erased, and its internal uses replaced with a local alloc.
// After scalrep, the store-to-load through the intermediate is forwarded.
//
// CHECK-LABEL: func.func @test_intermediate_internalization
// The intermediate operand should be removed; only one input remains.
// CHECK: d2m.generic
// CHECK: ins(%{{.*}} : memref<{{.*}}>)
// The intermediate remote_store and remote_load should be eliminated by scalrep.
// Only the input remote_load and output remote_store should remain.
// CHECK: d2m.remote_load
// CHECK: test.use
// CHECK: d2m.remote_store
// CHECK-NOT: d2m.remote_store
func.func @test_intermediate_internalization(
    %input: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  %intermediate = memref.alloc() : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %output = memref.alloc() : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  d2m.generic {block_factors = [1, 1], d2m.affine_fused, grid = #ttcore.grid<2x4>,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
    threads = [#d2m.thread<unified>]}
      ins(%input, %intermediate : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%output : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb_inter: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb_out: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    affine.for %i = 0 to %bf0 {
      affine.for %j = 0 to %bf1 {
        %idx0 = d2m.block_index(0) : index
        %idx1 = d2m.block_index(1) : index
        // Producer: load from input, store to intermediate
        %buf_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded = d2m.remote_load %buf_in %input[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %buf_mid = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored_inter = d2m.remote_store %intermediate[%idx0, %idx1] %buf_mid : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
        // Consumer: load from intermediate, compute, store to output
        %buf_in2 = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %loaded2 = d2m.remote_load %buf_in2 %intermediate[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        "test.use"(%loaded2) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> ()
        %buf_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %stored_out = d2m.remote_store %output[%idx0, %idx1] %buf_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }
  return
}
