// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Topology: an SFPU integer reduction (`d2m.tile_sfpu_reduce_sum`) whose
// result is consumed by a fused unary op (`d2m.tile_negative`) inside the
// same generic region.  The SFPU reduce declares
// `getNumDstScratchSlices() == 1`, so the dst-access insertion pass
// reserves a private scratch slot for it; the fused unary then asks for
// an in-place DST slot via `getCurrSliceIndex()`.  Scratch must live in
// its own pool: the negate must read/write the reduce's output slot
// (slot 0) and the reduce's `dst_scratch_index` must point at the
// separately-allocated slot 1.
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access-scheduled --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // ----------------------------------------------------------------------
  // Scheduled fallback path (flat block).  SFPU int reduce's accumulator
  // operand (`c`, operand index 1) is loaded into DST slot 0; scratch
  // lives in slot 1.  The fused `tile_negative` consumes the reduce's
  // result in place on slot 0.
  // ----------------------------------------------------------------------
  // CHECK-LABEL: func.func @fused_negate_after_int_reduce_scheduled
  func.func @fused_negate_after_int_reduce_scheduled(
      %in0: memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %init: memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out0: memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %init : memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>>
      %cb2_raw = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      %cb1 = d2m.wait %cb1_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      %cb2 = d2m.reserve %cb2_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      %a = affine.load %cb0[0, 0] : memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      %c = affine.load %cb1[0, 0] : memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      %r = "d2m.tile_sfpu_reduce_sum"(%a, %c) {reduce_dim = #d2m<reduce_dim R>} : (!ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
      %neg = "d2m.tile_negative"(%r) : (!ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
      affine.store %neg, %cb2[0, 0] : memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
    }
    // si32 -> 4-tile DST capacity.
    // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, si32>, #dst>
    //
    // CB->DST copy for the reduce's accumulator (operand index 1, slot 0):
    // CHECK: affine.store %{{.*}}, %[[DST]][0]
    //
    // Reduce reads its accumulator from DST slot 0 and gets scratch slot 1
    // (separately allocated):
    // CHECK: affine.load %[[DST]][0]
    // CHECK: %{{.*}} = "d2m.tile_sfpu_reduce_sum"({{.*}}) <{dst_scratch_index = 1 : i64, reduce_dim = #d2m<reduce_dim R>}>
    //
    // Reduce's output stored back to slot 0; the fused `tile_negative`
    // reads/writes the reduce's output slot (slot 0), not the scratch
    // slot (slot 1):
    // CHECK: affine.store %{{.*}}, %[[DST]][0]
    // CHECK: affine.load %[[DST]][0]
    // CHECK: "d2m.tile_negative"
    // CHECK-NOT: %[[DST]][1]
    // CHECK: affine.store %{{.*}}, %[[DST]][0]
    //
    // DST->CB store reads from slot 0:
    // CHECK: affine.load %[[DST]][0]
    // CHECK: affine.store
    return
  }

  // CHECK-LABEL: func.func @sfpu_reduce_scratch_index_linearized_under_loops
  func.func @sfpu_reduce_scratch_index_linearized_under_loops(
      %in0: memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %init: memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out0: memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %init : memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, si32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>>
      %cb2_raw = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      %cb1 = d2m.wait %cb1_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      %cb2 = d2m.reserve %cb2_raw : !d2m.cb<memref<1x1x!ttcore.tile<32x32, si32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
      affine.for %i = 1 to 5 step 2 {
        affine.for %j = 2 to 8 step 3 {
          %a = affine.load %cb0[0, 0] : memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
          %c = affine.load %cb1[0, 0] : memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
          %r = "d2m.tile_sfpu_reduce_sum"(%a, %c) {reduce_dim = #d2m<reduce_dim R>} : (!ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
          affine.store %r, %cb2[0, 0] : memref<1x1x!ttcore.tile<32x32, si32>, #l1_>
        }
      }
    }
    // Scratch slice 1 is linearized by the 2x2 enclosing affine loop trip-count
    // footprint, so the stamped index is 4 rather than the raw scratch slice 1.
    // CHECK: "d2m.tile_sfpu_reduce_sum"({{.*}}) <{dst_scratch_index = 4 : i64, reduce_dim = #d2m<reduce_dim R>}>
    return
  }

}
