// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc %s | FileCheck %s
//
// Comprehensive test for the hoisted loop structure created by graph coloring DST allocator.
// Verifies that the pass creates THREE separate loop nests:
//   1. Prologue loop nest - L1->DST data copies
//   2. Compute loop nest - Original computation using DST values
//   3. Epilogue loop nest - DST->L1 result copies

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @test_hoisted_loop_structure
  func.func @test_hoisted_loop_structure(
    %in0: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>,
    %in1: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>,
    %out: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>
  ) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 :
          memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>,
          memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>)
      outs(%out : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f16>, #l1_>

      // Original compute loop that loads from L1, computes, and stores to L1
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          %v0 = affine.load %mem0[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
          %v1 = affine.load %mem1[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
          %result = "d2m.tile_add"(%v0, %v1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          affine.store %result, %mem_out[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
        }
      }
    }
    return
  }
}

// CHECK: d2m.generic
// CHECK: ^compute0(%[[CB0:.*]]: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1>>, %[[CB1:.*]]: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1>>, %[[CB_OUT:.*]]: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1>>):

// Acquire DST at the beginning
// CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<3x2x2x!ttcore.tile<32x32, f16>, #dst>

// Wait/reserve operations
// CHECK: %[[MEM0:.*]] = d2m.wait %[[CB0]]
// CHECK: %[[MEM1:.*]] = d2m.wait %[[CB1]]
// CHECK: %[[MEM_OUT:.*]] = d2m.reserve %[[CB_OUT]]

// PARTIAL LOOP HOISTING: The pass fuses the outer loop and hoists copies to inner loops
// This creates a single outer loop with three sequential inner loops inside it

// CHECK: affine.for %[[OUTER_I:.*]] = 0 to 2 {

// PROLOGUE INNER LOOP: L1 -> DST copies for both inputs
// CHECK: affine.for %[[PROLOGUE_J:.*]] = 0 to 2 {
// CHECK: %[[L1_VAL0:.*]] = affine.load %[[MEM0]][%[[OUTER_I]], %[[PROLOGUE_J]]]
// CHECK: affine.store %[[L1_VAL0]], %[[DST]][{{.*}}, %[[OUTER_I]], %[[PROLOGUE_J]]]
// CHECK: %[[L1_VAL1:.*]] = affine.load %[[MEM1]][%[[OUTER_I]], %[[PROLOGUE_J]]]
// CHECK: affine.store %[[L1_VAL1]], %[[DST]][{{.*}}, %[[OUTER_I]], %[[PROLOGUE_J]]]
// CHECK: }

// COMPUTE INNER LOOP: Original computation using DST values
// CHECK: affine.for %[[COMPUTE_J:.*]] = 0 to 2 {
// CHECK: %[[DST_VAL0:.*]] = affine.load %[[DST]][{{.*}}, {{.*}}, {{.*}}]
// CHECK: %[[DST_VAL1:.*]] = affine.load %[[DST]][{{.*}}, {{.*}}, {{.*}}]
// CHECK: %[[COMPUTE_RESULT:.*]] = "d2m.tile_add"(%[[DST_VAL0]], %[[DST_VAL1]])
// CHECK: affine.store %[[COMPUTE_RESULT]], %[[DST]][{{.*}}, {{.*}}, {{.*}}]
// CHECK: }

// EPILOGUE INNER LOOP: DST -> L1 result copies
// CHECK: affine.for %[[EPILOGUE_J:.*]] = 0 to 2 {
// CHECK: %[[DST_RESULT:.*]] = affine.load %[[DST]][{{.*}}, %[[OUTER_I]], %[[EPILOGUE_J]]]
// CHECK: affine.store %[[DST_RESULT]], %[[MEM_OUT]][%[[OUTER_I]], %[[EPILOGUE_J]]]
// CHECK: }

// End of outer loop
// CHECK: }

// Release DST at the end
// CHECK: d2m.release_dst %[[DST]]
