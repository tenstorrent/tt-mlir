// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access='allocation-strategy=basic' --canonicalize %s | FileCheck %s --check-prefix=BASIC
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access='allocation-strategy=greedy' --canonicalize %s | FileCheck %s --check-prefix=GREEDY
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access='allocation-strategy=chaitin-briggs' --canonicalize %s | FileCheck %s --check-prefix=CHAITIN

// This test verifies the fix for the polygamma DST interference bug.
// The bug: greedy and chaitin-briggs incorrectly reused DST slot 0 for both
// tile_recip operations, causing the second recip to overwrite the first
// before tile_add could consume both values.
//
// Pattern that was failing:
//   %val0 = affine.load %subview[...]   // Load first input
//   %val1 = affine.load %subview_99[...] // Load second input
//   %recip1 = tile_recip(%val1)         // Compute first recip
//   STORE recip1 to DST slot 0
//   %recip0 = tile_recip(%val0)         // Compute second recip
//   STORE recip0 to DST slot 0          // BUG: overwrites recip1!
//   %result = tile_add(%recip0, %recip1)  // tile_add needs both values
//
// Fix: The interference graph must recognize that both tile_recip results
// are live when tile_add executes, so they need different DST slots.

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // The compute loop has two tile_recip operations. The key invariant is that
  // their results must be stored to DIFFERENT DST slots so tile_add can read both.
  //
  // For BASIC, the slots should be 0 and 1:
  // BASIC-LABEL: func.func @two_recip_then_add
  // BASIC: %[[DST:.*]] = d2m.acquire_dst()
  // BASIC: "d2m.tile_recip"
  // BASIC-NEXT: affine.store %{{.*}}, %[[DST]][0,
  // BASIC: "d2m.tile_recip"
  // BASIC-NEXT: affine.store %{{.*}}, %[[DST]][1,
  //
  // For GREEDY, the bug was that both used slot 0. After fix, they should differ:
  // GREEDY-LABEL: func.func @two_recip_then_add
  // GREEDY: %[[DST:.*]] = d2m.acquire_dst()
  // GREEDY: "d2m.tile_recip"
  // GREEDY-NEXT: affine.store %{{.*}}, %[[DST]][0,
  // GREEDY: "d2m.tile_recip"
  // GREEDY-NEXT: affine.store %{{.*}}, %[[DST]][1,
  //
  // For CHAITIN, same as GREEDY - the bug was identical:
  // CHAITIN-LABEL: func.func @two_recip_then_add
  // CHAITIN: %[[DST:.*]] = d2m.acquire_dst()
  // CHAITIN: "d2m.tile_recip"
  // CHAITIN-NEXT: affine.store %{{.*}}, %[[DST]][0,
  // CHAITIN: "d2m.tile_recip"
  // CHAITIN-NEXT: affine.store %{{.*}}, %[[DST]][1,
  
  func.func @two_recip_then_add(
      %in0: memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %in1: memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out: memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<4x4>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 :
          memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
          memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out : memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
                               -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
                               -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
                                       -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %subview0 = memref.subview %mem0[%arg1, %arg2] [1, 1] [1, 1]
              : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
              to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview1 = memref.subview %mem1[%arg1, %arg2] [1, 1] [1, 1]
              : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
              to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview_out = memref.subview %mem_out[%arg1, %arg2] [1, 1] [1, 1]
              : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
              to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          affine.for %arg3 = 0 to 1 {
            affine.for %arg4 = 0 to 1 {
              %val0 = affine.load %subview0[%arg3, %arg4]
                  : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
              %val1 = affine.load %subview1[%arg3, %arg4]
                  : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
              // Two tile_recip operations whose results both feed tile_add.
              // These must use DIFFERENT DST slots because tile_add needs both
              // values simultaneously.
              %recip1 = "d2m.tile_recip"(%val1)
                  : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %recip0 = "d2m.tile_recip"(%val0)
                  : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %result = "d2m.tile_add"(%recip0, %recip1)
                  : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              affine.store %result, %subview_out[%arg3, %arg4]
                  : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
            }
          } {d2m.linalg_root}
        }
      }
    }
    return
  }
}
