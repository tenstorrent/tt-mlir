// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access="use-tile-matmul=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>
module {
  func.func @eltwise_unary_chain_multi_tile(%in0: memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #l1_>,
                                            %out0: memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #l1_>)
        outs(%out0 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #l1_>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>, %cb1: memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.for %arg1 = %c0 to %c4 step %c2 {
        scf.for %arg2 = %c0 to %c4 step %c4 {
          %subview = memref.subview %cb0[%arg1, %arg2] [2, 4] [1, 1] : memref<4x4x!ttcore.tile<32x32, bf16>, #l1_> to memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #l1_>
          %subview_4 = memref.subview %cb1[%arg1, %arg2] [2, 4] [1, 1] : memref<4x4x!ttcore.tile<32x32, bf16>, #l1_> to memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #l1_>
          %dst = d2m.acquire_dst() : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 4 {
              %0 = affine.load %subview[%arg3, %arg4] : memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #l1_>
              affine.store %0, %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
            }
          }
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 4 {
              %0 = affine.load %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              %1 = "d2m.tile_abs"(%0) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: %[[ABS_RESULT:.*]] = "d2m.tile_abs"(%[[DST0_VAL:.*]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: affine.store %[[ABS_RESULT]], %dst[%c0, %[[ARG_I:.*]], %[[ARG_J:.*]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              affine.store %1, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              %2 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              // CHECK: %[[DST_ABS:.*]] = affine.load %dst[%c0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              %3 = "d2m.tile_sin"(%2) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: %[[SIN_RESULT:.*]] = "d2m.tile_sin"(%[[DST_ABS]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: affine.store %[[SIN_RESULT]], %dst[%c0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              affine.store %3, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              %4 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              // CHECK: %[[DST_SIN:.*]] = affine.load %dst[%c0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              %5 = "d2m.tile_negative"(%4) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: %[[NEG_RESULT:.*]] = "d2m.tile_negative"(%[[DST_SIN]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: affine.store %[[NEG_RESULT]], %dst[%c0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              affine.store %5, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              %6 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              // CHECK: %[[DST_NEG:.*]] = affine.load %dst[%c0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              %7 = "d2m.tile_exp"(%6) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: %[[EXP_RESULT:.*]] = "d2m.tile_exp"(%[[DST_NEG]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: affine.store %[[EXP_RESULT]], %dst[0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              affine.store %7, %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
            }
          }
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 4 {
              %0 = affine.load %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              // CHECK: %[[FINAL_VAL:.*]] = affine.load %dst[0, %[[ARG_I:.*]], %[[ARG_J:.*]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              // CHECK: affine.store %[[FINAL_VAL]], %subview_{{[0-9]*}}[%[[ARG_I]], %[[ARG_J]]] : memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #l1>
              affine.store %0, %subview_4[%arg3, %arg4] : memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #l1_>
            }
          }
        }
      }
    }
    return
  }
}
