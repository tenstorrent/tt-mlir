// RUN: ttmlir-opt %s -d2m-insert-dst-register-gc -split-input-file | FileCheck %s --check-prefix=CHECK

// Test: Large DST allocation with bf16 - verify release_dst is inserted correctly.
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @large_dst_eltwise
// CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<16x4x4x!ttcore.tile<32x32, bf16>, #dst{{.*}}>
// CHECK: affine.for
// CHECK: affine.store
// CHECK: affine.load
// CHECK: "d2m.tile_add"
// CHECK: "d2m.tile_mul"
// CHECK: affine.load
// CHECK: affine.store
// CHECK: d2m.release_dst %[[DST]]
// CHECK: return
func.func @large_dst_eltwise(%cb0: memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>,
                             %cb1: memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>,
                             %cb2: memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>) {
  %c0 = arith.constant 0 : index

  %dst = d2m.acquire_dst() : memref<16x4x4x!ttcore.tile<32x32, bf16>, #dst_>

  // Load inputs
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %cb0[%i, %j] : memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>
      affine.store %0, %dst[0, %i, %j] : memref<16x4x4x!ttcore.tile<32x32, bf16>, #dst_>

      %1 = affine.load %cb1[%i, %j] : memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>
      affine.store %1, %dst[1, %i, %j] : memref<16x4x4x!ttcore.tile<32x32, bf16>, #dst_>
    }
  }

  // Compute: add then multiply
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %dst[0, %i, %j] : memref<16x4x4x!ttcore.tile<32x32, bf16>, #dst_>
      %1 = affine.load %dst[1, %i, %j] : memref<16x4x4x!ttcore.tile<32x32, bf16>, #dst_>
      %2 = "d2m.tile_add"(%0, %1) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      affine.store %2, %dst[2, %i, %j] : memref<16x4x4x!ttcore.tile<32x32, bf16>, #dst_>

      %3 = affine.load %dst[2, %i, %j] : memref<16x4x4x!ttcore.tile<32x32, bf16>, #dst_>
      %4 = "d2m.tile_mul"(%3, %3) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      affine.store %4, %dst[3, %i, %j] : memref<16x4x4x!ttcore.tile<32x32, bf16>, #dst_>
    }
  }

  // Writeback
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %dst[3, %i, %j] : memref<16x4x4x!ttcore.tile<32x32, bf16>, #dst_>
      affine.store %0, %cb2[%i, %j] : memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>
    }
  }

  return
}

// -----

// Test: Large DST matmul pattern - verify release_dst is inserted correctly.
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @large_dst_matmul
// CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<8x4x4x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHECK: affine.for
// CHECK: affine.store
// CHECK: affine.for
// CHECK: affine.for
// CHECK: "d2m.tile_matmul"
// CHECK: affine.store
// CHECK: affine.for
// CHECK: affine.load
// CHECK: affine.store
// CHECK: d2m.release_dst %[[DST]]
// CHECK: return
func.func @large_dst_matmul(%cb0: memref<4x4x!ttcore.tile<32x32, f32>, #l1_>,
                            %cb1: memref<4x4x!ttcore.tile<32x32, f32>, #l1_>,
                            %cb2: memref<4x4x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index

  %dst = d2m.acquire_dst() : memref<8x4x4x!ttcore.tile<32x32, f32>, #dst_>

  // Initialize accumulator
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %cb2[%i, %j] : memref<4x4x!ttcore.tile<32x32, f32>, #l1_>
      affine.store %0, %dst[0, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, f32>, #dst_>
    }
  }

  // Matmul computation
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      affine.for %k = 0 to 4 {
        %a = affine.load %cb0[%i, %k] : memref<4x4x!ttcore.tile<32x32, f32>, #l1_>
        %b = affine.load %cb1[%k, %j] : memref<4x4x!ttcore.tile<32x32, f32>, #l1_>
        %acc = affine.load %dst[0, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, f32>, #dst_>
        %result = "d2m.tile_matmul"(%a, %b, %acc) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %result, %dst[0, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, f32>, #dst_>
      }
    }
  }

  // Writeback
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %dst[0, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, f32>, #dst_>
      affine.store %0, %cb2[%i, %j] : memref<4x4x!ttcore.tile<32x32, f32>, #l1_>
    }
  }

  return
}
