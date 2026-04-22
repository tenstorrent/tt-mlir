// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for the scheduled pass (d2m-insert-dst-register-access-scheduled)
// on hand-crafted IR with d2m.scheduled attribute on the root loop.
// Exercises:
//   - In-place CB->DST load insertion (dataCopyGenerateScheduledInPlace)
//   - Cloned loop skeleton for DST->CB stores (dataCopyGenerateWithClone)
//   - DstStackAllocator (stack-based allocation)
//   - Intermediate results through DST
//   - Consumption of d2m.scheduled attribute
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access-scheduled --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // -----------------------------------------------------------------------
  // Unary eltwise with d2m.scheduled:
  //   - Single input CB -> single DST slot
  //   - In-place CB->DST load, compute in DST, DST->CB store (cloned loop)
  //   - d2m.scheduled consumed after processing
  // -----------------------------------------------------------------------
  // CHECK-LABEL: func.func @unary_scheduled
  func.func @unary_scheduled(
      %in0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>,
      %out0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.reserve %cb1_raw : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst>
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 4 {
          %v = affine.load %cb0[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          %r = "d2m.tile_exp"(%v) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %r, %cb1[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        }
      } {d2m.linalg_root, d2m.scheduled}
      // d2m.scheduled should be consumed (removed)
      // CHECK-NOT: d2m.scheduled
      // In-place CB->DST load + compute loop:
      // CHECK: affine.for %[[I:.*]] = 0 to 2
      // CHECK:   affine.for %[[J:.*]] = 0 to 4
      // CHECK:     %[[CB_VAL:.*]] = affine.load %{{.*}}[%[[I]], %[[J]]]
      // CHECK:     affine.store %[[CB_VAL]], %[[DST]]
      // CHECK:     %[[DST_VAL:.*]] = affine.load %[[DST]]
      // CHECK:     %[[EXP:.*]] = "d2m.tile_exp"(%[[DST_VAL]])
      // CHECK:     affine.store %[[EXP]], %[[DST]]
      // Cloned store loop (DST->CB):
      // CHECK: affine.for %{{.*}} = 0 to 2
      // CHECK:   affine.for %{{.*}} = 0 to 4
      // CHECK:     %[[STORE_VAL:.*]] = affine.load %[[DST]]
      // CHECK:     affine.store %[[STORE_VAL]], %{{.*}}
    }
    return
  }

  // -----------------------------------------------------------------------
  // Binary eltwise with d2m.scheduled (3 inputs for OpScheduler to trigger):
  //   - add(in0, in1) -> div(result, in2) -> store out
  //   - DstStackAllocator allocates slots for each input
  //   - Intermediate result (add) stored in DST
  //   - Separate cloned loop for DST->CB stores
  // -----------------------------------------------------------------------
  // CHECK-LABEL: func.func @binary_chain_scheduled
  func.func @binary_chain_scheduled(
      %in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %in2: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %in1, %in2 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb2_raw = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb3_raw = d2m.get_cb(3) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.wait %cb1_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb2 = d2m.wait %cb2_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb3 = d2m.reserve %cb3_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %sv0 = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %sv1 = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %sv2 = memref.subview %cb2[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %sv3 = memref.subview %cb3[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst>
      affine.for %i = 0 to 1 {
        affine.for %j = 0 to 1 {
          %a = affine.load %sv0[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %b = affine.load %sv1[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %c = affine.load %sv2[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %add = "d2m.tile_add"(%a, %b) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          %div = "d2m.tile_div"(%add, %c) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %div, %sv3[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
        }
      } {d2m.linalg_root, d2m.scheduled}
      // d2m.scheduled consumed
      // CHECK-NOT: d2m.scheduled
      // DST access inserted with stack allocator; CB loads → DST, compute in DST:
      // CHECK: affine.for
      // CHECK:   affine.for
      // CHECK:     affine.load %{{.*}} : memref<1x1x!ttcore.tile<32x32, f32>
      // CHECK:     affine.store %{{.*}}, %[[DST]]
      // CHECK:     affine.load %[[DST]]
      // CHECK:     "d2m.tile_add"
      // CHECK:     affine.store %{{.*}}, %[[DST]]
      // CHECK:     "d2m.tile_div"
      // CHECK:     affine.store %{{.*}}, %[[DST]]
      // Separate cloned store loop (DST -> CB):
      // CHECK: affine.for
      // CHECK:   affine.for
      // CHECK:     affine.load %[[DST]]
      // CHECK:     affine.store %{{.*}}, %{{.*}} : memref<1x1x!ttcore.tile<32x32, f32>
    }
    return
  }

  // -----------------------------------------------------------------------
  // Chain of 3 ops with intermediates through DST (scheduled):
  //   add(in0, in1) -> recip(add_result) -> div(recip_result, in2) -> store
  //   Exercises the dstIntermediates map in the stack allocator.
  //   Uses f16 to get a larger DST capacity (8 tiles).
  // -----------------------------------------------------------------------
  // CHECK-LABEL: func.func @chain_intermediates_scheduled
  func.func @chain_intermediates_scheduled(
      %in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>,
      %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>,
      %in2: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>,
      %out0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %in1, %in2 : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>
      %cb2_raw = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>
      %cb3_raw = d2m.get_cb(3) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %cb1 = d2m.wait %cb1_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %cb2 = d2m.wait %cb2_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %cb3 = d2m.reserve %cb3_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %c0 = arith.constant 0 : index
      %sv0 = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
      %sv1 = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
      %sv2 = memref.subview %cb2[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
      %sv3 = memref.subview %cb3[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
      // f16 -> 8-tile DST capacity
      // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<8x!ttcore.tile<32x32, f16>, #dst>
      affine.for %i = 0 to 1 {
        affine.for %j = 0 to 1 {
          %a = affine.load %sv0[%i, %j] : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
          %b = affine.load %sv1[%i, %j] : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
          %c = affine.load %sv2[%i, %j] : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
          %add = "d2m.tile_add"(%a, %b) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %recip = "d2m.tile_recip"(%add) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %div = "d2m.tile_div"(%recip, %c) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          affine.store %div, %sv3[%i, %j] : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
        }
      } {d2m.linalg_root, d2m.scheduled}
      // CHECK-NOT: d2m.scheduled
      // Stack allocator: add loads 2 tiles, intermediate through recip, div uses 3rd input
      // CHECK: affine.for
      // CHECK:   affine.for
      //            CB->DST load for in2 (the third operand, used by div):
      // CHECK:     affine.load %{{.*}} : memref<1x1x!ttcore.tile<32x32, f16>
      // CHECK:     affine.store %{{.*}}, %[[DST]]
      // CHECK:     affine.load %[[DST]]
      //            tile_add intermediate (uses direct CB loads for first two operands):
      // CHECK:     "d2m.tile_add"
      // CHECK:     affine.store %{{.*}}, %[[DST]]
      // CHECK:     affine.load %[[DST]]
      //            tile_recip (reads intermediate from DST):
      // CHECK:     "d2m.tile_recip"
      // CHECK:     affine.store %{{.*}}, %[[DST]]
      // CHECK:     affine.load %[[DST]]
      //            tile_div (reads recip result and in2 from DST):
      // CHECK:     "d2m.tile_div"
      // CHECK:     affine.store %{{.*}}, %[[DST]]
      // Cloned store loop:
      // CHECK: affine.for
      // CHECK:   affine.for
      // CHECK:     affine.load %[[DST]]
      // CHECK:     affine.store %{{.*}}, %{{.*}} : memref<1x1x!ttcore.tile<32x32, f16>
    }
    return
  }
}
