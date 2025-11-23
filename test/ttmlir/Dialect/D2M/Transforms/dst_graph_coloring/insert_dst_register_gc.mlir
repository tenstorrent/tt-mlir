// RUN: ttmlir-opt %s -d2m-insert-dst-register-gc -split-input-file | FileCheck %s --check-prefix=CHECK

// Test 1: Simple pass-through test.
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @test_pass_simple
// CHECK: %{{.*}} = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHECK: d2m.release_dst
// CHECK: return
func.func @test_pass_simple() {
  %dst = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

// Test 3: Real D2M IR with acquire_dst (from insert_dst_register_access tests).
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @eltwise_binary_with_dst
// CHECK: %{{.*}} = d2m.acquire_dst() : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHECK: affine.load
// CHECK: affine.store %{{.*}}, %{{.*}}[0
// CHECK: affine.load
// CHECK: affine.store %{{.*}}, %{{.*}}[1
// CHECK: affine.load %{{.*}}[0
// CHECK: affine.load %{{.*}}[1
// CHECK: "d2m.tile_add"
// CHECK: affine.store %{{.*}}, %{{.*}}[2
// CHECK: affine.load %{{.*}}[2
// CHECK: affine.store %{{.*}}, %arg2
// CHECK: d2m.release_dst
func.func @eltwise_binary_with_dst(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>,
                                    %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>,
                                    %cb2: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) {
  %dst = d2m.acquire_dst() : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %c0 = arith.constant 0 : index

  %0 = affine.load %cb0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %0, %dst[0, %c0, %c0] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>

  %1 = affine.load %cb1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %1, %dst[1, %c0, %c0] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>

  %2 = affine.load %dst[0, %c0, %c0] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %3 = affine.load %dst[1, %c0, %c0] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %4 = "d2m.tile_add"(%2, %3) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
  affine.store %4, %dst[2, %c0, %c0] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>

  %5 = affine.load %dst[2, %c0, %c0] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
  affine.store %5, %cb2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

  d2m.release_dst %dst : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

// Test 4: Multiple DST registers in same function (simulating complex scenario).
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @multiple_dst_regions
// CHECK: %{{.*}} = d2m.acquire_dst() : memref<2x2x2x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHECK-COUNT-2: affine.for %{{.*}} = 0 to 2
// CHECK: %{{.*}} = d2m.acquire_dst() : memref<2x2x2x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHECK: affine.for %{{.*}} = 0 to 2
// CHECK: affine.for %{{.*}} = 0 to 2
// CHECK: d2m.release_dst
// CHECK: d2m.release_dst
func.func @multiple_dst_regions(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>,
                                %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dst1 = d2m.acquire_dst() : memref<2x2x2x!ttcore.tile<32x32, f32>, #dst_>
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 2 {
      %0 = affine.load %cb0[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      affine.store %0, %dst1[0, %i, %j] : memref<2x2x2x!ttcore.tile<32x32, f32>, #dst_>
    }
  }

  %dst2 = d2m.acquire_dst() : memref<2x2x2x!ttcore.tile<32x32, f32>, #dst_>
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 2 {
      %0 = affine.load %cb1[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      affine.store %0, %dst2[0, %i, %j] : memref<2x2x2x!ttcore.tile<32x32, f32>, #dst_>
    }
  }

  d2m.release_dst %dst1 : memref<2x2x2x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst2 : memref<2x2x2x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

// Test 5: DST with reduction (matmul pattern simulation).
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @dst_with_accumulation
// CHECK: %{{.*}} = d2m.acquire_dst() : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHECK: affine.for %{{.*}} = 0 to 3
// CHECK: affine.store %{{.*}}, %{{.*}}[0
// CHECK: affine.for %{{.*}} = 0 to 3
// CHECK: affine.store %{{.*}}, %{{.*}}[1
// CHECK: affine.for %{{.*}} = 0 to 3
// CHECK: affine.for %{{.*}} = 0 to 2
// CHECK: affine.store %{{.*}}, %{{.*}}[2
// CHECK-COUNT-3: affine.load %{{.*}}[
// CHECK: "d2m.tile_matmul"
// CHECK: affine.store %{{.*}}, %{{.*}}[2
// CHECK: d2m.release_dst
func.func @dst_with_accumulation(%cb0: memref<3x3x!ttcore.tile<32x32, f32>, #l1_>,
                                 %cb1: memref<3x2x!ttcore.tile<32x32, f32>, #l1_>,
                                 %cb2: memref<3x2x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dst = d2m.acquire_dst() : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst_>

  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 3 {
      %0 = affine.load %cb0[%i, %j] : memref<3x3x!ttcore.tile<32x32, f32>, #l1_>
      affine.store %0, %dst[0, %i, %j] : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst_>
    }
  }

  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 2 {
      %0 = affine.load %cb1[%i, %j] : memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
      affine.store %0, %dst[1, %i, %j] : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst_>
    }
  }

  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 2 {
      %0 = affine.load %cb2[%i, %j] : memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
      affine.store %0, %dst[2, %i, %j] : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst_>

      %1 = affine.load %dst[0, %i, %c0] : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst_>
      %2 = affine.load %dst[1, %i, %j] : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst_>
      %3 = affine.load %dst[2, %i, %j] : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst_>
      %4 = "d2m.tile_matmul"(%1, %2, %3) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      affine.store %4, %dst[2, %i, %j] : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst_>
    }
  }

  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 2 {
      %0 = affine.load %dst[2, %i, %j] : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst_>
      affine.store %0, %cb2[%i, %j] : memref<3x2x!ttcore.tile<32x32, f32>, #l1_>
    }
  }

  d2m.release_dst %dst : memref<3x3x2x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

// Test 6: Unary operation with DST (in-place pattern from existing tests).
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @unary_in_place
// CHECK: %{{.*}} = d2m.acquire_dst() : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.load
// CHECK: affine.store %{{.*}}, %{{.*}}[0
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.load %{{.*}}[0
// CHECK: "d2m.tile_exp"
// CHECK: affine.store %{{.*}}, %{{.*}}[0
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.load %{{.*}}[0
// CHECK: affine.store %{{.*}}, %arg1
// CHECK: d2m.release_dst
func.func @unary_in_place(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
                          %cb1: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index

  %dst = d2m.acquire_dst() : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>

  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %cb0[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      affine.store %0, %dst[0, %i, %j] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
    }
  }

  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %dst[0, %i, %j] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
      %1 = "d2m.tile_exp"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      affine.store %1, %dst[0, %i, %j] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
    }
  }

  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %dst[0, %i, %j] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
      affine.store %0, %cb1[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
  }

  d2m.release_dst %dst : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

// Test 7: Large DST allocation (from insert_dst_register_access_eltwise_large_dst)
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @eltwise_large_dst
// CHECK: d2m.acquire_dst
// CHECK: affine.store
// CHECK: affine.load
// CHECK: "d2m.tile_add"
// CHECK: "d2m.tile_mul"
// CHECK: d2m.release_dst
func.func @eltwise_large_dst(%cb0: memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>,
                             %cb1: memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>,
                             %cb2: memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>) {
  %c0 = arith.constant 0 : index

  %dst = d2m.acquire_dst() : memref<8x4x4x!ttcore.tile<32x32, bf16>, #dst_>

  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %cb0[%i, %j] : memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>
      affine.store %0, %dst[0, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, bf16>, #dst_>

      %1 = affine.load %cb1[%i, %j] : memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>
      affine.store %1, %dst[1, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, bf16>, #dst_>
    }
  }

  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %dst[0, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, bf16>, #dst_>
      %1 = affine.load %dst[1, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, bf16>, #dst_>
      %2 = "d2m.tile_add"(%0, %1) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      affine.store %2, %dst[2, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, bf16>, #dst_>

      %3 = affine.load %dst[2, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, bf16>, #dst_>
      %4 = "d2m.tile_mul"(%3, %3) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      affine.store %4, %dst[3, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, bf16>, #dst_>
    }
  }

  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %0 = affine.load %dst[3, %i, %j] : memref<8x4x4x!ttcore.tile<32x32, bf16>, #dst_>
      affine.store %0, %cb2[%i, %j] : memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>
    }
  }

  d2m.release_dst %dst : memref<8x4x4x!ttcore.tile<32x32, bf16>, #dst_>
  return
}
