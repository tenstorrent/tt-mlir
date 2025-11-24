// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc --canonicalize %s | FileCheck %s --check-prefix=LEGACY
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=chaitin-briggs" --canonicalize %s | FileCheck %s --check-prefix=CHAITIN
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc="coloring-strategy=greedy" --canonicalize %s | FileCheck %s --check-prefix=GREEDY
//
// Integration tests for graph coloring on elementwise operations.
// Tests input that already has DST infrastructure (acquire_dst, affine loops, tile operations).
// The new pass applies graph coloring to optimize DST register allocation.

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // LEGACY-LABEL: func.func @binary
  // LEGACY: %[[DST0:.*]] = d2m.acquire_dst
  // LEGACY-NEXT: affine.for %
  // LEGACY: affine.store
  // LEGACY: d2m.tile_maximum
  // LEGACY: d2m.release_dst %[[DST0]]
  // LEGACY-NEXT: return

  // CHAITIN-LABEL: func.func @binary
  // CHAITIN: %[[DST0:.*]] = d2m.acquire_dst
  // CHAITIN-NEXT: affine.for %
  // CHAITIN: affine.store
  // CHAITIN: d2m.tile_maximum
  // CHAITIN: d2m.release_dst %[[DST0]]
  // CHAITIN-NEXT: return

  // GREEDY-LABEL: func.func @binary
  // GREEDY: %[[DST0:.*]] = d2m.acquire_dst
  // GREEDY-NEXT: affine.for %
  // GREEDY: affine.store
  // GREEDY: d2m.tile_maximum
  // GREEDY: d2m.release_dst %[[DST0]]
  // GREEDY-NEXT: return

  func.func @binary(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                    %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                    %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    %dst = d2m.acquire_dst() : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 1 {
        %0 = affine.load %in0[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
        affine.store %0, %dst[0, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %1 = affine.load %in1[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
        affine.store %1, %dst[1, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
      }
    }
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 1 {
        %0 = affine.load %dst[0, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %1 = affine.load %dst[1, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %2 = "d2m.tile_maximum"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %2, %dst[2, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
      }
    }
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 1 {
        %0 = affine.load %dst[2, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
        affine.store %0, %out0[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
      }
    }
    d2m.release_dst %dst : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
    return
  }

  // LEGACY-LABEL: func.func @ternary_with_interference_and_reuse
  // LEGACY: %[[DST0:.*]] = d2m.acquire_dst
  // LEGACY-NEXT: affine.for %
  // LEGACY: d2m.tile_maximum
  // LEGACY: d2m.release_dst %[[DST0]]
  // LEGACY-NEXT: %[[DST1:.*]] = d2m.acquire_dst
  // LEGACY: d2m.tile_add
  // LEGACY: d2m.release_dst %[[DST1]]
  // LEGACY-NEXT: return

  // CHAITIN-LABEL: func.func @ternary_with_interference_and_reuse
  // CHAITIN: %[[DST0:.*]] = d2m.acquire_dst
  // CHAITIN-NEXT: affine.for %
  // CHAITIN: d2m.tile_maximum
  // CHAITIN: d2m.release_dst %[[DST0]]
  // CHAITIN-NEXT: %[[DST1:.*]] = d2m.acquire_dst
  // CHAITIN: d2m.tile_add
  // CHAITIN: d2m.release_dst %[[DST1]]
  // CHAITIN-NEXT: return

  // GREEDY-LABEL: func.func @ternary_with_interference_and_reuse
  // GREEDY: %[[DST0:.*]] = d2m.acquire_dst
  // GREEDY-NEXT: affine.for %
  // GREEDY: d2m.tile_maximum
  // GREEDY: d2m.release_dst %[[DST0]]
  // GREEDY-NEXT: %[[DST1:.*]] = d2m.acquire_dst
  // GREEDY: d2m.tile_add
  // GREEDY: d2m.release_dst %[[DST1]]
  // GREEDY-NEXT: return

  func.func @ternary_with_interference_and_reuse(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                                       %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                                       %in2: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                                       %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    %dst0 = d2m.acquire_dst() : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 1 {
        %0 = affine.load %in0[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
        affine.store %0, %dst0[0, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %1 = affine.load %in1[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
        affine.store %1, %dst0[1, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %2 = affine.load %in2[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
        affine.store %2, %dst0[2, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %r3a = affine.load %dst0[0, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %r3b = affine.load %dst0[1, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %r3 = "d2m.tile_maximum"(%r3a, %r3b) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %r3, %dst0[3, %i, %j] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
      }
    }
    d2m.release_dst %dst0 : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst_>
    %dst1 = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 1 {
        %load2 = affine.load %in2[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
        affine.store %load2, %dst1[0, %i, %j] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
      }
    }
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 1 {
        %loaded = affine.load %dst1[0, %i, %j] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %load0 = affine.load %in0[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
        affine.store %load0, %dst1[1, %i, %j] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
        %result = "d2m.tile_add"(%loaded, %load0) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %result, %out0[0, 0, %i, %j] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>
      }
    }
    d2m.release_dst %dst1 : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
    return
  }
}
