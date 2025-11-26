// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-access --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefixes=CHECK,LEGACY
// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-gc="coloring-strategy=greedy" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,GREEDY
// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-gc="coloring-strategy=chaitin-briggs" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,CHAITIN

// Test for intermediate compute operations in chains.
// When a compute operation's result is consumed by another compute operation
// (not stored to L1), the intermediate value is materialized in DST registers
// via store+load pairs. With loop-indexed DST accesses, operations have
// disjoint live ranges and naturally share DST slots through graph coloring.

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @intermediate_compute
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<{{[0-9]+}}x1x1x!ttcore.tile<32x32, f32>,
  // CHECK: affine.for
  // LEGACY: d2m.tile_sub{{.*}}result_dst_index = 2
  // GREEDY: d2m.tile_sub{{.*}}result_dst_index = 2
  // CHAITIN: d2m.tile_sub{{.*}}result_dst_index = 2
  // CHECK:   affine.store
  // CHECK:   affine.load
  // LEGACY: d2m.tile_nez{{.*}}result_dst_index = 2
  // GREEDY: d2m.tile_nez{{.*}}result_dst_index = 0
  // CHAITIN: d2m.tile_nez{{.*}}result_dst_index = 2
  // CHECK:   affine.store

  func.func @intermediate_compute(
      %in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1 {
        scf.for %arg1 = %c0 to %c1 step %c1 {
          %subview = memref.subview %cb0[%arg0, %arg1] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview_1 = memref.subview %cb1[%arg0, %arg1] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview_2 = memref.subview %cb2[%arg0, %arg1] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %in_17: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %0 = "d2m.tile_sub"(%in, %in_17) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            %1 = "d2m.tile_nez"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %1 : !ttcore.tile<32x32, f32>
          }
        }
      }
    }
    return
  }
}
