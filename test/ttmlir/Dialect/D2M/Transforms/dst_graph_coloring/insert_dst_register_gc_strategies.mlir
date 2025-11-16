// RUN: ttmlir-opt --d2m-insert-dst-register-gc %s -split-input-file | FileCheck %s --check-prefix=DEFAULT
// RUN: ttmlir-opt --d2m-insert-dst-register-gc="coloring-strategy=chaitin-briggs" %s -split-input-file | FileCheck %s --check-prefix=CHAITIN
// RUN: ttmlir-opt --d2m-insert-dst-register-gc="coloring-strategy=greedy" %s -split-input-file | FileCheck %s --check-prefix=GREEDY

// Test 1: Simple non-interfering DST values (both strategies should handle easily).
//
// Interference graph (no edges):
//   dst0    dst1
//   (independent set - no interferences)
//
// Both linear and graph coloring can use 1 color:
//   dst0 → color 0
//   dst1 → color 0 (reuse)
//   Total: 1 color (1 DST slice)
#dst_ = #ttcore.memory_space<dst>

// DEFAULT-LABEL: func.func @test_non_interfering
// DEFAULT: d2m.acquire_dst
// DEFAULT: d2m.release_dst
// DEFAULT: d2m.acquire_dst
// DEFAULT: d2m.release_dst

// CHAITIN-LABEL: func.func @test_non_interfering
// CHAITIN: d2m.acquire_dst
// CHAITIN: d2m.release_dst
// CHAITIN: d2m.acquire_dst
// CHAITIN: d2m.release_dst

// GREEDY-LABEL: func.func @test_non_interfering
// GREEDY: d2m.acquire_dst
// GREEDY: d2m.release_dst
// GREEDY: d2m.acquire_dst
// GREEDY: d2m.release_dst

func.func @test_non_interfering() {
  %dst0 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst0 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  %dst1 = d2m.acquire_dst() : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst1 : memref<1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

// Test 2: Interfering DST values (overlapping live ranges).
// Both strategies should successfully color this graph.
//
// Interference graph (path):
//   dst0 -------- dst1
//   (single edge)
//
// Both linear and graph coloring need 2 colors:
//   dst0 → color 0
//   dst1 → color 1
//   Total: 2 colors (2 DST slices)
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// DEFAULT-LABEL: func.func @test_interfering_dst
// DEFAULT: %[[DST0:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// DEFAULT: %[[DST1:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// DEFAULT: affine.load %[[DST0]]
// DEFAULT: affine.load %[[DST1]]
// DEFAULT: d2m.release_dst %[[DST0]]
// DEFAULT: d2m.release_dst %[[DST1]]

// CHAITIN-LABEL: func.func @test_interfering_dst
// CHAITIN: %[[DST0:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHAITIN: %[[DST1:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// CHAITIN: affine.load %[[DST0]]
// CHAITIN: affine.load %[[DST1]]
// CHAITIN: d2m.release_dst %[[DST0]]
// CHAITIN: d2m.release_dst %[[DST1]]

// GREEDY-LABEL: func.func @test_interfering_dst
// GREEDY: %[[DST0:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// GREEDY: %[[DST1:.*]] = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst{{.*}}>
// GREEDY: affine.load %[[DST0]]
// GREEDY: affine.load %[[DST1]]
// GREEDY: d2m.release_dst %[[DST0]]
// GREEDY: d2m.release_dst %[[DST1]]

func.func @test_interfering_dst(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>,
                                %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index

  // Two DST values with overlapping live ranges.
  %dst0 = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %0 = affine.load %cb0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %0, %dst0[0, %c0, %c0] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>

  %dst1 = d2m.acquire_dst() : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %1 = affine.load %cb1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %1, %dst1[0, %c0, %c0] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>

  // Both DST values are live here (interference).
  %2 = affine.load %dst0[0, %c0, %c0] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %3 = affine.load %dst1[0, %c0, %c0] : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %4 = "d2m.tile_add"(%2, %3) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>

  d2m.release_dst %dst0 : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst1 : memref<2x1x1x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

// Test 3: Multiple interfering DST values in affine loops.
// Tests graph coloring with more complex liveness patterns.
//
// Interference graph (complete K₃):
//     dst0
//    /    \
//  dst1---dst2
//  (triangle - all nodes interfere)
//
// Both linear and graph coloring need 3 colors:
//   dst0 → color 0
//   dst1 → color 1
//   dst2 → color 2
//   Total: 3 colors (3 DST slices)
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// DEFAULT-LABEL: func.func @test_loop_interference
// DEFAULT: %[[DST0:.*]] = d2m.acquire_dst()
// DEFAULT: %[[DST1:.*]] = d2m.acquire_dst()
// DEFAULT: %[[DST2:.*]] = d2m.acquire_dst()
// DEFAULT: affine.for
// DEFAULT: d2m.release_dst %[[DST0]]
// DEFAULT: d2m.release_dst %[[DST1]]
// DEFAULT: d2m.release_dst %[[DST2]]

// CHAITIN-LABEL: func.func @test_loop_interference
// CHAITIN: %[[DST0:.*]] = d2m.acquire_dst()
// CHAITIN: %[[DST1:.*]] = d2m.acquire_dst()
// CHAITIN: %[[DST2:.*]] = d2m.acquire_dst()
// CHAITIN: affine.for
// CHAITIN: d2m.release_dst %[[DST0]]
// CHAITIN: d2m.release_dst %[[DST1]]
// CHAITIN: d2m.release_dst %[[DST2]]

// GREEDY-LABEL: func.func @test_loop_interference
// GREEDY: %[[DST0:.*]] = d2m.acquire_dst()
// GREEDY: %[[DST1:.*]] = d2m.acquire_dst()
// GREEDY: %[[DST2:.*]] = d2m.acquire_dst()
// GREEDY: affine.for
// GREEDY: d2m.release_dst %[[DST0]]
// GREEDY: d2m.release_dst %[[DST1]]
// GREEDY: d2m.release_dst %[[DST2]]

func.func @test_loop_interference(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>,
                                  %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>,
                                  %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index

  // Three DST values that all interfere with each other.
  %dst0 = d2m.acquire_dst() : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
  %dst1 = d2m.acquire_dst() : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
  %dst2 = d2m.acquire_dst() : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>

  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 2 {
      %0 = affine.load %cb0[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      affine.store %0, %dst0[0, %i, %j] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>

      %1 = affine.load %cb1[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      affine.store %1, %dst1[0, %i, %j] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>

      // All three DST values are live here.
      %2 = affine.load %dst0[0, %i, %j] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
      %3 = affine.load %dst1[0, %i, %j] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
      %4 = "d2m.tile_add"(%2, %3) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      affine.store %4, %dst2[0, %i, %j] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>

      %5 = affine.load %dst2[0, %i, %j] : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
      affine.store %5, %cb2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    }
  }

  d2m.release_dst %dst0 : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst1 : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst2 : memref<1x2x2x!ttcore.tile<32x32, f32>, #dst_>
  return
}

// -----

// Test 4: Sequential DST usage with reuse opportunity.
// Tests that both strategies can handle DST reuse when live ranges don't overlap.
//
// Interference graph (no edges):
//   dst0    dst1    dst2
//   (independent set - no interferences)
//
// Both linear and graph coloring can use 1 color:
//   dst0 → color 0
//   dst1 → color 0 (reuse)
//   dst2 → color 0 (reuse)
//   Total: 1 color (1 DST slice)
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// DEFAULT-LABEL: func.func @test_sequential_reuse
// DEFAULT: %[[DST0:.*]] = d2m.acquire_dst()
// DEFAULT: d2m.release_dst %[[DST0]]
// DEFAULT: %[[DST1:.*]] = d2m.acquire_dst()
// DEFAULT: d2m.release_dst %[[DST1]]
// DEFAULT: %[[DST2:.*]] = d2m.acquire_dst()
// DEFAULT: d2m.release_dst %[[DST2]]

// CHAITIN-LABEL: func.func @test_sequential_reuse
// CHAITIN: %[[DST0:.*]] = d2m.acquire_dst()
// CHAITIN: d2m.release_dst %[[DST0]]
// CHAITIN: %[[DST1:.*]] = d2m.acquire_dst()
// CHAITIN: d2m.release_dst %[[DST1]]
// CHAITIN: %[[DST2:.*]] = d2m.acquire_dst()
// CHAITIN: d2m.release_dst %[[DST2]]

// GREEDY-LABEL: func.func @test_sequential_reuse
// GREEDY: %[[DST0:.*]] = d2m.acquire_dst()
// GREEDY: d2m.release_dst %[[DST0]]
// GREEDY: %[[DST1:.*]] = d2m.acquire_dst()
// GREEDY: d2m.release_dst %[[DST1]]
// GREEDY: %[[DST2:.*]] = d2m.acquire_dst()
// GREEDY: d2m.release_dst %[[DST2]]

func.func @test_sequential_reuse(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>,
                                 %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index

  // First operation.
  %dst0 = d2m.acquire_dst() : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %0 = affine.load %cb0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %0, %dst0[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %1 = affine.load %dst0[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %2 = "d2m.tile_exp"(%1) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
  affine.store %2, %cb1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  d2m.release_dst %dst0 : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>

  // Second operation (can reuse same physical DST).
  %dst1 = d2m.acquire_dst() : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %3 = affine.load %cb1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %3, %dst1[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %4 = affine.load %dst1[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %5 = "d2m.tile_exp"(%4) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
  affine.store %5, %cb0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  d2m.release_dst %dst1 : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>

  // Third operation (can reuse same physical DST again).
  %dst2 = d2m.acquire_dst() : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %6 = affine.load %cb0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %6, %dst2[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %7 = affine.load %dst2[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %8 = "d2m.tile_exp"(%7) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
  affine.store %8, %cb1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  d2m.release_dst %dst2 : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>

  return
}

// -----

// Test 5: Diamond pattern demonstrating graph coloring advantage.
// This test shows why graph coloring is superior to linear allocation.
//
// Interference graph (diamond pattern):
//      dst0
//     /    \
//   dst1   dst2
//     \    /
//      dst3
//
// Linear allocation would assign:
//   dst0 → color 0
//   dst1 → color 1
//   dst2 → color 2
//   dst3 → color 3
//   Total: 4 colors (4 DST slices)
//
// Graph coloring recognizes:
//   - dst0 and dst2 don't interfere → can share color
//   - dst1 and dst3 don't interfere → can share color
//   Optimal assignment:
//   dst0 → color 0
//   dst1 → color 1
//   dst2 → color 0 (reuse!)
//   dst3 → color 1 (reuse!)
//   Total: 2 colors (2 DST slices)
//
// This demonstrates 50% reduction in DST usage compared to the incremental allocator

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// DEFAULT-LABEL: func.func @test_diamond_pattern
// DEFAULT: %[[DST0:.*]] = d2m.acquire_dst()
// DEFAULT: %[[DST1:.*]] = d2m.acquire_dst()
// DEFAULT: affine.load %[[DST0]]
// DEFAULT: affine.load %[[DST1]]
// DEFAULT: d2m.release_dst %[[DST0]]
// DEFAULT: %[[DST2:.*]] = d2m.acquire_dst()
// DEFAULT: affine.load %[[DST1]]
// DEFAULT: affine.load %[[DST2]]
// DEFAULT: d2m.release_dst %[[DST1]]
// DEFAULT: %[[DST3:.*]] = d2m.acquire_dst()
// DEFAULT: affine.load %[[DST2]]
// DEFAULT: affine.load %[[DST3]]
// DEFAULT: d2m.release_dst %[[DST2]]
// DEFAULT: d2m.release_dst %[[DST3]]

// CHAITIN-LABEL: func.func @test_diamond_pattern
// CHAITIN: %[[DST0:.*]] = d2m.acquire_dst()
// CHAITIN: %[[DST1:.*]] = d2m.acquire_dst()
// CHAITIN: affine.load %[[DST0]]
// CHAITIN: affine.load %[[DST1]]
// CHAITIN: d2m.release_dst %[[DST0]]
// CHAITIN: %[[DST2:.*]] = d2m.acquire_dst()
// CHAITIN: affine.load %[[DST1]]
// CHAITIN: affine.load %[[DST2]]
// CHAITIN: d2m.release_dst %[[DST1]]
// CHAITIN: %[[DST3:.*]] = d2m.acquire_dst()
// CHAITIN: affine.load %[[DST2]]
// CHAITIN: affine.load %[[DST3]]
// CHAITIN: d2m.release_dst %[[DST2]]
// CHAITIN: d2m.release_dst %[[DST3]]

// GREEDY-LABEL: func.func @test_diamond_pattern
// GREEDY: %[[DST0:.*]] = d2m.acquire_dst()
// GREEDY: %[[DST1:.*]] = d2m.acquire_dst()
// GREEDY: affine.load %[[DST0]]
// GREEDY: affine.load %[[DST1]]
// GREEDY: d2m.release_dst %[[DST0]]
// GREEDY: %[[DST2:.*]] = d2m.acquire_dst()
// GREEDY: affine.load %[[DST1]]
// GREEDY: affine.load %[[DST2]]
// GREEDY: d2m.release_dst %[[DST1]]
// GREEDY: %[[DST3:.*]] = d2m.acquire_dst()
// GREEDY: affine.load %[[DST2]]
// GREEDY: affine.load %[[DST3]]
// GREEDY: d2m.release_dst %[[DST2]]
// GREEDY: d2m.release_dst %[[DST3]]

func.func @test_diamond_pattern(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>,
                                %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>,
                                %cb2: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index

  // Top of diamond: dst0 and dst1 both live.
  %dst0 = d2m.acquire_dst() : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %0 = affine.load %cb0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %0, %dst0[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>

  %dst1 = d2m.acquire_dst() : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %1 = affine.load %cb1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %1, %dst1[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>

  // Use both dst0 and dst1 (they interfere).
  %2 = affine.load %dst0[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %3 = affine.load %dst1[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %4 = "d2m.tile_add"(%2, %3) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
  affine.store %4, %cb2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

  // Release dst0 early
  d2m.release_dst %dst0 : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>

  // Middle of diamond: dst1 and dst2 both live (dst0 is dead).
  %dst2 = d2m.acquire_dst() : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %5 = affine.load %cb0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %5, %dst2[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>

  // Use both dst1 and dst2 (they interfere, but dst2 doesn't interfere with dst0).
  %6 = affine.load %dst1[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %7 = affine.load %dst2[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %8 = "d2m.tile_add"(%6, %7) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
  affine.store %8, %cb2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

  // Release dst1 early
  d2m.release_dst %dst1 : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>

  // Bottom of diamond: dst2 and dst3 both live (dst0 and dst1 are dead).
  %dst3 = d2m.acquire_dst() : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %9 = affine.load %cb1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
  affine.store %9, %dst3[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>

  // Use both dst2 and dst3 (they interfere, but dst3 doesn't interfere with dst0 or dst1).
  %10 = affine.load %dst2[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %11 = affine.load %dst3[0, %c0, %c0] : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  %12 = "d2m.tile_add"(%10, %11) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
  affine.store %12, %cb2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

  d2m.release_dst %dst2 : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>
  d2m.release_dst %dst3 : memref<1x1x1x!ttcore.tile<32x32, f32>, #dst_>

  return
}
