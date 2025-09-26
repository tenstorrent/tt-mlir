// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access="use-tile-matmul=false" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
module {
  func.func @eq(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048>, #l1_>,
                %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048>, #l1_>,
                %out0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048>, #l1_>)  {
    ^compute0(%cb0: memref<1x1x!ttcore.tile<32x32, f16>, #l1_>, %cb1: memref<1x1x!ttcore.tile<32x32, f16>, #l1_>, %cb2: memref<1x1x!ttcore.tile<32x32, f16>, #l1_>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1 {
        scf.for %arg1 = %c0 to %c1 step %c1 {
          %subview = memref.subview %cb0[%arg0, %arg1] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
          %subview_1 = memref.subview %cb1[%arg0, %arg1] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
          %subview_2 = memref.subview %cb2[%arg0, %arg1] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>) {
          ^bb0(%in: !ttcore.tile<32x32, f16>, %in_17: !ttcore.tile<32x32, f16>, %out: !ttcore.tile<32x32, f16>):
            %0 = "d2m.tile_div"(%in, %in_17) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
            // CHECK: %[[DIV_RESULT:.*]] = "d2m.tile_div"(%[[DST0_VAL:.*]], %[[DST1_VAL:.*]]) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
            // CHECK: affine.store %[[DIV_RESULT]], %[[DST:.*]][2, %[[ARG_I:.*]], %[[ARG_J:.*]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
            // CHECK: %[[DST_DIV:.*]] = affine.load %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
            %1 = "d2m.tile_recip"(%0) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
             // CHECK: %[[RECIP_RESULT:.*]] = "d2m.tile_recip"(%[[DST_DIV]]) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
            // CHECK: affine.store %[[RECIP_RESULT]], %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
            // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
            // CHECK: affine.store %[[FINAL_VAL]], %cb2[%[[ARG_I]], %[[ARG_J]]] : memref<1x1x!ttcore.tile<32x32, f16>, #l1>
            linalg.yield %1 : !ttcore.tile<32x32, f16>
          }
        }
      }
    }
    return
  }
}
