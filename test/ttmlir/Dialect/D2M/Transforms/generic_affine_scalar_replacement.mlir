// RUN: ttmlir-opt --allow-unregistered-dialect --d2m-generic-affine-scalrep="enable=true" %s | FileCheck %s

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
// After scalar replacement, the generic op should still be present with the
// block_index operations.
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
// top-level users). Scalar replacement should still simplify the load/store
// sequence while preserving valid remote traffic in the fused generic body.
//
// CHECK-LABEL: func.func @test_intermediate_internalization
// The fused generic is preserved and still contains the expected remote ops.
// CHECK: d2m.generic
// CHECK: ins(%{{.*}} : memref<{{.*}}>)
// One input load remains.
// CHECK: d2m.remote_load
// Store to the intermediate followed by compute and final output store remain.
// CHECK: d2m.remote_store
// CHECK: test.use
// CHECK: d2m.remote_store
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

// Test 7: Complex matmul + add fusion with multiple block factors and indices.
// Tests scalar replacement on nested affine loops with reduction dimension.
// The intermediate (%alloc) is internalized and the store-to-load through it
// is fully scalar replaced. Generic-local intermediate allocs are lowered to
// d2m.scratch_allocate at the top of the unified region. Remote loads for
// matmul inputs, compute (matmul, add), and final store should be preserved.
//
// CHECK-LABEL: func.func @test_matmul_add_subset_fusion
// CHECK: d2m.generic
// CHECK: d2m.scratch_allocate
// CHECK: d2m.get_block_factor
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.for
// CHECK: d2m.block_index
// CHECK: d2m.remote_load
// CHECK: d2m.tile_matmul_block
// CHECK: linalg.generic
// CHECK: d2m.tile_add
// CHECK: d2m.remote_store
func.func @test_matmul_add_subset_fusion(
    %arg0: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
    %arg1: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
    %arg2: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  %alloc = memref.alloc() : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
  %alloc_0 = memref.alloc() : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

  d2m.generic {block_factors = [1, 1, 2], d2m.affine_fused, grid = #ttcore.grid<2x2>,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>],
    threads = [#d2m.thread<unified>]}
      ins(%arg0, %arg1, %alloc : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
      outs(%alloc_0 : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
  ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb3: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    %block_factor0 = d2m.get_block_factor(0) : index
    %block_factor1 = d2m.get_block_factor(1) : index
    %block_factor2 = d2m.get_block_factor(2) : index
    affine.for %i0 = 0 to %block_factor0 {
      affine.for %i1 = 0 to %block_factor1 {
        %accum_buf = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        affine.for %i2 = 0 to %block_factor2 {
          %block0 = d2m.block_index(0) : index
          %block1 = d2m.block_index(1) : index
          %block2 = d2m.block_index(2) : index
          %buf_a = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %buf_b = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %loaded_a = d2m.remote_load %buf_a %arg0[%block0, %block2] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %loaded_b = d2m.remote_load %buf_b %arg1[%block2, %block1] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          "d2m.tile_matmul_block"(%loaded_a, %loaded_b, %accum_buf) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
        } {d2m.blocking_loop = 2 : i64}
        %block0_out = d2m.block_index(0) : index
        %block1_out = d2m.block_index(1) : index
        %stored_inter = d2m.remote_store %alloc[%block0_out, %block1_out] %accum_buf : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
        %buf_bias = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        %loaded_bias = d2m.remote_load %buf_bias %alloc[%block0_out, %block1_out] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        %buf_result = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
          ins(%loaded_bias, %loaded_bias : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
          outs(%buf_result : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) {
        ^bb0(%in0: !ttcore.tile<32x32, f32>, %in1: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %sum = "d2m.tile_add"(%in0, %in1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %sum : !ttcore.tile<32x32, f32>
        }
        %stored = d2m.remote_store %alloc_0[%block0_out, %block1_out] %buf_result : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
      } {d2m.blocking_loop = 1 : i64}
    } {d2m.blocking_loop = 0 : i64}
  }
  return
}

// Test 8: Ensure temporary block_offset-to-constant bridge is reversed.
// The pass may use placeholder arith.constant values internally, but the
// resulting IR must keep d2m.block_offset and must not leak tag attrs.
//
// CHECK-LABEL: func.func @test_block_offset_bridge_roundtrip
// CHECK: d2m.generic
// CHECK: d2m.block_offset
// CHECK-NOT: d2m.block_offset_dim
// CHECK: d2m.remote_load
// CHECK: d2m.remote_store
func.func @test_block_offset_bridge_roundtrip(
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
        %off0 = d2m.block_offset(0) : index
        %idx0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%i)[%off0]
        %off1 = d2m.block_offset(1) : index
        %idx1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%j)[%off1]
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
