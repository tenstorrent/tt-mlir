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
