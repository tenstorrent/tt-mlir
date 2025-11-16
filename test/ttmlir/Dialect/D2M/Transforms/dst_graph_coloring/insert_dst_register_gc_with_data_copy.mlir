// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-gc --split-input-file %s | FileCheck %s
//
// Verifies that the d2m-insert-dst-register-gc pass does the following:
//   1. acquire_dst is created
//   2. L1→DST copy loops are generated before operations
//   3. Original loads are replaced with DST loads
//   4. DST→L1 copy loops are generated for results
//   5. Graph coloring assigns optimal slice indices avoiding overwrites (v0→slice 2, v1→slice 1, result→slice 0)

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @test_linalg_input
  // CHECK: d2m.generic
  // CHECK: ^compute0(%[[CB0:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>, %[[CB1:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>, %[[CB_OUT:.*]]: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1>>):

  // With corrected liveness analysis, a single add needs 3 slices:
  // - 1 slice for v0 (live from load until used in tile_add)
  // - 1 slice for v1 (live from load until used in tile_add)
  // - 1 slice for result (live from tile_add until store)
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<3x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index

  // CHECK: %[[MEM0:.*]] = d2m.wait %[[CB0]]
  // CHECK-NEXT: %[[MEM1:.*]] = d2m.wait %[[CB1]]
  // CHECK-NEXT: %[[MEM_OUT:.*]] = d2m.reserve %[[CB_OUT]]

  // Load from L1 input 0, store to DST slice 2, then load from DST
  // Graph coloring assigns slice 2 to v0 because it's live during the add operation
  // CHECK: %[[L1_VAL0:.*]] = affine.load %[[MEM0]][%[[C0]], %[[C0]]]
  // CHECK-NEXT: affine.store %[[L1_VAL0]], %[[DST]][2, %[[C0]], %[[C0]]]
  // CHECK-NEXT: %[[DST_VAL0:.*]] = affine.load %[[DST]][2, %[[C0]], %[[C0]]]

  // Load from L1 input 1, store to DST slice 1, then load from DST
  // Graph coloring assigns slice 1 to v1 (different from v0 since both are live during add)
  // CHECK: %[[L1_VAL1:.*]] = affine.load %[[MEM1]][%[[C0]], %[[C0]]]
  // CHECK-NEXT: affine.store %[[L1_VAL1]], %[[DST]][1, %[[C0]], %[[C0]]]
  // CHECK-NEXT: %[[DST_VAL1:.*]] = affine.load %[[DST]][1, %[[C0]], %[[C0]]]

  // Compute operation uses values loaded from DST
  // CHECK: %[[RESULT:.*]] = "d2m.tile_add"(%[[DST_VAL0]], %[[DST_VAL1]])

  // Store result to DST slice 0, load from DST, then store to L1 output
  // Graph coloring can reuse a slice for the result since v0 and v1 are no longer live
  // CHECK-NEXT: affine.store %[[RESULT]], %[[DST]][0, %[[C0]], %[[C0]]]
  // CHECK-NEXT: %[[DST_RESULT:.*]] = affine.load %[[DST]][0, %[[C0]], %[[C0]]]
  // CHECK-NEXT: affine.store %[[DST_RESULT]], %[[MEM_OUT]][%[[C0]], %[[C0]]]

  // CHECK: d2m.release_dst %[[DST]]

  func.func @test_linalg_input(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
                                %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
                                %out: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
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
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
      %c0 = arith.constant 0 : index

      %0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %out_mem = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

      %v0 = affine.load %0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %v1 = affine.load %1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %result = "d2m.tile_add"(%v0, %v1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>

      affine.store %result, %out_mem[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
    }
    return
  }
}

// -----

// Verifies that the pass leaves linalg.generic operations untransformed.
// The GC pass should not convert linalg ops - that's a separate concern.

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @test_with_linalg
  // CHECK: d2m.generic
  // CHECK: linalg.generic
  // CHECK-NOT: d2m.acquire_dst
  // CHECK-NOT: affine.store

  func.func @test_with_linalg(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
                               %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
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
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
          memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>)
      outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %c0 = arith.constant 0 : index
      %subview0 = memref.subview %mem0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
      %subview1 = memref.subview %mem1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
      %subview_out = memref.subview %mem_out[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
      linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%subview0, %subview1 : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>,
                                   memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>)
        outs(%subview_out : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f16>, %arg1: !ttcore.tile<32x32, f16>, %arg2: !ttcore.tile<32x32, f16>):
        %result = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
        linalg.yield %result : !ttcore.tile<32x32, f16>
      }
    }
    return
  }
}
