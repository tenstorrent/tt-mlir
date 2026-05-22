// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access-unscheduled --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Tests L1 accumulation guard insertion in the unscheduled pass (with the
// default disable-l1-acc=false, i.e. L1 accumulation enabled):
//
// 1. `matmul_l1_acc`: with a K-block reduction loop wrapping a
//    tile_matmul root, expect d2m.set_l1_accumulate to receive a flag
//    derived from the K-block IV (the closest ancestor whose IV the output
//    store does NOT depend on).
//
// 2. `matmul_l1_acc_nested_parallel_scratch`: when a scratch output store
//    makes an outer parallel loop look reduction-like, still use the closest
//    K-block reduction loop as the trigger.
//
// 3. `matmul_no_outer_reduction`: when the only ancestor loops of
//    acquire_dst are *parallel* (output indexed by them), L1-acc must
//    NOT be enabled. Using the outermost loop unconditionally is
//    incorrect for cases like batched matmuls where the outermost loop
//    is parallel.

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @matmul_l1_acc
  func.func @matmul_l1_acc(
      %in0: memref<1x1x4x4x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x2048, 1>, #l1_>,
      %in1: memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>,
      %out0: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %in1 : memref<1x1x4x4x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x2048, 1>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f16>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>>
      %cb2_raw = d2m.get_cb(2) : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f16>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, f16>, #l1_>
      %cb1 = d2m.wait %cb1_raw : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f16>, #l1_>
      %cb2 = d2m.reserve %cb2_raw : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      // Outer K-block reduction loop with 2 iterations (K0=0, K0=2). Note:
      // the output (sv2 = subview of cb2) does NOT depend on %k_block, so
      // this is the correct reduction loop to use as the L1-acc trigger.
      scf.for %k_block = %c0 to %c4 step %c2 {
        %sv0 = memref.subview %cb0[%c0, %k_block] [2, 2] [1, 1] : memref<4x4x!ttcore.tile<32x32, f16>, #l1_> to memref<2x2x!ttcore.tile<32x32, f16>, strided<[4, 1], offset: ?>, #l1_>
        %sv1 = memref.subview %cb1[%k_block, %c0] [2, 2] [1, 1] : memref<4x2x!ttcore.tile<32x32, f16>, #l1_> to memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
        %sv2 = memref.subview %cb2[%c0, %c0] [2, 2] [1, 1] : memref<2x2x!ttcore.tile<32x32, f16>, #l1_> to memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
        affine.for %i = 0 to 2 {
          affine.for %j = 0 to 2 {
            affine.for %k = 0 to 2 {
              %a = affine.load %sv0[%i, %k] : memref<2x2x!ttcore.tile<32x32, f16>, strided<[4, 1], offset: ?>, #l1_>
              %b = affine.load %sv1[%k, %j] : memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
              %c = affine.load %sv2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
              %r = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
              affine.store %r, %sv2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
            }
          }
        } {d2m.linalg_root}
      }
    }
    return
  }
  // acquire_dst placed inside the K-block scf.for:
  // CHECK: scf.for %[[K_BLOCK:.*]] = %c0
  // CHECK:   %[[DST:.*]] = d2m.acquire_dst() : memref<{{[0-9]+}}x!ttcore.tile<32x32, f16>, #dst>
  // L1 accumulation is explicitly disabled on the first K-block iteration and
  // enabled on later K-block iterations. The flag must be reset this way
  // because the packer setting persists across outer output-tile iterations.
  // CHECK:   %[[NOT_FIRST:.*]] = arith.cmpi ne, %[[K_BLOCK]], %c0
  // CHECK:   %[[L1_ACC_FLAG:.*]] = arith.extui %[[NOT_FIRST]] : i1 to i32
  // CHECK:   d2m.set_l1_accumulate(%[[L1_ACC_FLAG]])
  // Compute loop with matmul:
  // CHECK:   affine.for
  // CHECK:     affine.for
  // CHECK:       affine.for
  // CHECK:         "d2m.tile_matmul"

  // CHECK-LABEL: func.func @matmul_l1_acc_nested_parallel_scratch
  func.func @matmul_l1_acc_nested_parallel_scratch(
      %in0: memref<1x1x4x4x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x2048, 1>, #l1_>,
      %in1: memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>,
      %out0: memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %in1 : memref<1x1x4x4x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x2048, 1>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f16>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f16>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, f16>, #l1_>
      %cb1 = d2m.wait %cb1_raw : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f16>, #l1_>
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.for %tile_j = %c0 to %c4 step %c2 {
        %scratch = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
        scf.for %k_block = %c0 to %c4 step %c2 {
          %sv0 = memref.subview %cb0[%c0, %k_block] [2, 2] [1, 1] : memref<4x4x!ttcore.tile<32x32, f16>, #l1_> to memref<2x2x!ttcore.tile<32x32, f16>, strided<[4, 1], offset: ?>, #l1_>
          %sv1 = memref.subview %cb1[%k_block, %c0] [2, 2] [1, 1] : memref<4x2x!ttcore.tile<32x32, f16>, #l1_> to memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
          affine.for %i = 0 to 2 {
            affine.for %j = 0 to 2 {
              affine.for %k = 0 to 2 {
                %a = affine.load %sv0[%i, %k] : memref<2x2x!ttcore.tile<32x32, f16>, strided<[4, 1], offset: ?>, #l1_>
                %b = affine.load %sv1[%k, %j] : memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
                %c = affine.load %scratch[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
                %r = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
                affine.store %r, %scratch[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, #l1_>
              }
            }
          } {d2m.linalg_root}
        }
      }
    }
    return
  }
  // CHECK: scf.for %[[TILE_J:.*]] = %c0
  // CHECK:   scf.for %[[K_BLOCK:.*]] = %c0
  // CHECK:     %[[DST:.*]] = d2m.acquire_dst()
  // CHECK-NOT: arith.cmpi eq, %[[TILE_J]]
  // CHECK:     %[[NOT_FIRST:.*]] = arith.cmpi ne, %[[K_BLOCK]], %c0
  // CHECK:     %[[L1_ACC_FLAG:.*]] = arith.extui %[[NOT_FIRST]] : i1 to i32
  // CHECK:     d2m.set_l1_accumulate(%[[L1_ACC_FLAG]])

  // CHECK-LABEL: func.func @matmul_no_outer_reduction
  func.func @matmul_no_outer_reduction(
      %in0: memref<1x1x4x4x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x2048, 1>, #l1_>,
      %in1: memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>,
      %out0: memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%in0, %in1 : memref<1x1x4x4x!ttcore.tile<32x32, f16>, #ttcore.shard<8192x2048, 1>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x4x2x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x2048, 1>, #l1_>) {
    ^unified0:
      %cb0_raw = d2m.get_cb(0) : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f16>, #l1_>>
      %cb1_raw = d2m.get_cb(1) : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>>
      %cb2_raw = d2m.get_cb(2) : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>>
      %cb0 = d2m.wait %cb0_raw : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f16>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, f16>, #l1_>
      %cb1 = d2m.wait %cb1_raw : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f16>, #l1_>
      %cb2 = d2m.reserve %cb2_raw : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f16>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f16>, #l1_>
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      // Outer parallel tiling loop over M positions: the output cb2 IS
      // indexed by %tile_i, so %tile_i is a *parallel* loop, not a
      // reduction loop. L1-acc must NOT be enabled here.
      scf.for %tile_i = %c0 to %c4 step %c2 {
        scf.for %tile_j = %c0 to %c2 step %c2 {
          %sv0 = memref.subview %cb0[%tile_i, %c0] [2, 4] [1, 1] : memref<4x4x!ttcore.tile<32x32, f16>, #l1_> to memref<2x4x!ttcore.tile<32x32, f16>, strided<[4, 1], offset: ?>, #l1_>
          %sv1 = memref.subview %cb1[%c0, %tile_j] [4, 2] [1, 1] : memref<4x2x!ttcore.tile<32x32, f16>, #l1_> to memref<4x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
          %sv2 = memref.subview %cb2[%tile_i, %tile_j] [2, 2] [1, 1] : memref<4x2x!ttcore.tile<32x32, f16>, #l1_> to memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
          affine.for %i = 0 to 2 {
            affine.for %j = 0 to 2 {
              affine.for %k = 0 to 4 {
                %a = affine.load %sv0[%i, %k] : memref<2x4x!ttcore.tile<32x32, f16>, strided<[4, 1], offset: ?>, #l1_>
                %b = affine.load %sv1[%k, %j] : memref<4x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
                %c = affine.load %sv2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
                %r = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
                affine.store %r, %sv2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f16>, strided<[2, 1], offset: ?>, #l1_>
              }
            }
          } {d2m.linalg_root}
        }
      }
    }
    return
  }
  // CHECK-NOT: d2m.set_l1_accumulate
}
