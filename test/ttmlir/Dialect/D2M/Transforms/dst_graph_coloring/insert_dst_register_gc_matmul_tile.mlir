// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc --canonicalize %s | FileCheck %s --check-prefix=DEFAULT
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=chaitin-briggs" --canonicalize %s | FileCheck %s --check-prefix=CHAITIN
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=greedy" --canonicalize %s | FileCheck %s --check-prefix=GREEDY
//
// Smoke tests for graph coloring on matmul operations.
// Tests that the pass successfully processes d2m.generic patterns with matmul
// using all three coloring strategies without errors.

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // DEFAULT-LABEL: func.func @matmul_simple
  // DEFAULT: %[[DST:.*]] = d2m.acquire_dst
  // DEFAULT-NEXT: affine.for
  // DEFAULT: affine.store
  // DEFAULT: d2m.tile_matmul
  // DEFAULT: affine.load
  // DEFAULT: affine.store
  // DEFAULT: d2m.release_dst %[[DST]]
  // DEFAULT-NEXT: return

  // CHAITIN-LABEL: func.func @matmul_simple
  // CHAITIN: %[[DST:.*]] = d2m.acquire_dst
  // CHAITIN-NEXT: affine.for
  // CHAITIN: affine.store
  // CHAITIN: d2m.tile_matmul
  // CHAITIN: affine.load
  // CHAITIN: affine.store
  // CHAITIN: d2m.release_dst %[[DST]]
  // CHAITIN-NEXT: return

  // GREEDY-LABEL: func.func @matmul_simple
  // GREEDY: %[[DST:.*]] = d2m.acquire_dst
  // GREEDY-NEXT: affine.for
  // GREEDY: affine.store
  // GREEDY: d2m.tile_matmul
  // GREEDY: affine.load
  // GREEDY: affine.store
  // GREEDY: d2m.release_dst %[[DST]]
  // GREEDY-NEXT: return

  func.func @matmul_simple(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                           %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                           %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    %dst = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst_>
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 1 {
        %a = affine.load %in0[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
        affine.store %a, %dst[0, %i, %j] : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %b = affine.load %in1[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
        affine.store %b, %dst[1, %i, %j] : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst_>
      }
    }
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 1 {
        %a = affine.load %dst[0, %i, %j] : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %b = affine.load %dst[1, %i, %j] : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %c = affine.load %out0[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
        affine.store %c, %dst[2, %i, %j] : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %result = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %result, %dst[2, %i, %j] : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst_>
      }
    }
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 1 {
        %result = affine.load %dst[2, %i, %j] : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst_>
        affine.store %result, %out0[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
      }
    }
    d2m.release_dst %dst : memref<3x1x1x!ttcore.tile<32x32, f32>, #dst_>
    return
  }
}
