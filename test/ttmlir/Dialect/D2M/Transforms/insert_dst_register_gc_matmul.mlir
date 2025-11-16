// RUN: ttmlir-opt %s -d2m-insert-dst-register-gc -split-input-file | FileCheck %s --check-prefix=CHECK

// Test: Matmul pattern with acquire_dst - verify release_dst is inserted correctly.
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @matmul_with_dst
// CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<1x3x2x!ttcore.tile<32x32, f16>, #dst{{.*}}>
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.store
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.load
// CHECK: affine.load
// CHECK: "d2m.tile_matmul"
// CHECK: affine.store
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.load
// CHECK: affine.store
// CHECK: d2m.release_dst %[[DST]]
// CHECK: return
func.func @matmul_with_dst(%cb0: memref<3x3x!ttcore.tile<32x32, f16>, #l1_>,
                            %cb1: memref<3x2x!ttcore.tile<32x32, f16>, #l1_>,
                            %cb2: memref<3x2x!ttcore.tile<32x32, f16>, #l1_>) {
  %c0 = arith.constant 0 : index

  %dst = d2m.acquire_dst() : memref<1x3x2x!ttcore.tile<32x32, f16>, #dst_>

  // Initialize accumulator from output buffer
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 2 {
      %0 = affine.load %cb2[%i, %j] : memref<3x2x!ttcore.tile<32x32, f16>, #l1_>
      affine.store %0, %dst[0, %i, %j] : memref<1x3x2x!ttcore.tile<32x32, f16>, #dst_>
    }
  }

  // Matmul computation
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 2 {
      affine.for %k = 0 to 3 {
        %a = affine.load %cb0[%i, %k] : memref<3x3x!ttcore.tile<32x32, f16>, #l1_>
        %b = affine.load %cb1[%k, %j] : memref<3x2x!ttcore.tile<32x32, f16>, #l1_>
        %acc = affine.load %dst[0, %i, %j] : memref<1x3x2x!ttcore.tile<32x32, f16>, #dst_>
        %result = "d2m.tile_matmul"(%a, %b, %acc) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
        affine.store %result, %dst[0, %i, %j] : memref<1x3x2x!ttcore.tile<32x32, f16>, #dst_>
      }
    }
  }

  // Writeback to output buffer
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 2 {
      %0 = affine.load %dst[0, %i, %j] : memref<1x3x2x!ttcore.tile<32x32, f16>, #dst_>
      affine.store %0, %cb2[%i, %j] : memref<3x2x!ttcore.tile<32x32, f16>, #l1_>
    }
  }

  return
}

// -----

// Test: Multiple matmul operations with interference - verify graph coloring works.
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @multiple_matmul_interference
// CHECK: %[[DST0:.*]] = d2m.acquire_dst() : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHECK: %[[DST1:.*]] = d2m.acquire_dst() : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHECK: affine.for
// CHECK: "d2m.tile_matmul"
// CHECK: affine.for
// CHECK: "d2m.tile_matmul"
// CHECK: d2m.release_dst %[[DST0]]
// CHECK: d2m.release_dst %[[DST1]]
// CHECK: return
func.func @multiple_matmul_interference(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>,
                                        %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>,
                                        %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>,
                                        %cb3: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index

  %dst0 = d2m.acquire_dst() : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
  %dst1 = d2m.acquire_dst() : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>

  // First matmul
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 2 {
      %a = affine.load %cb0[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %b = affine.load %cb1[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %acc = affine.load %dst0[0, %i, %j] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
      %result = "d2m.tile_matmul"(%a, %b, %acc) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      affine.store %result, %dst0[0, %i, %j] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
    }
  }

  // Second matmul (interferes with first)
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 2 {
      %a = affine.load %cb2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %b = affine.load %cb3[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %acc = affine.load %dst1[0, %i, %j] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
      %result = "d2m.tile_matmul"(%a, %b, %acc) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      affine.store %result, %dst1[0, %i, %j] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
    }
  }

  return
}
