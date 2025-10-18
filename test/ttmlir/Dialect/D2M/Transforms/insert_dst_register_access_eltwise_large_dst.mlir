// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access="max-dst-physical-size-tiles=32" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @binary
  func.func @binary(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                    %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                    %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    ^compute0(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<1x1x!ttcore.tile<32x32, f32>, #l1_>):
      %c0 = arith.constant 0 : index
      %subview = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_1 = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_2 = memref.subview %cb2[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
        // Check that the operands are stored to dst memory space
        // CHECK: %[[ARG0_VAL:.*]] = affine.load %[[ARG0:.*]]
        // CHECK: affine.store %[[ARG0_VAL]], %[[DST]]
        // CHECK: %[[ARG1_VAL:.*]] = affine.load %[[ARG1:.*]]
        // CHECK: affine.store %[[ARG1_VAL]], %[[DST]]
        // CHECK: %[[DST0_VAL:.*]] = affine.load %[[DST]]
        // CHECK: %[[DST1_VAL:.*]] = affine.load %[[DST]]
        // CHECK: %[[MAXIMUM_RESULT:.*]] = "d2m.tile_maximum"(%[[DST0_VAL]], %[[DST1_VAL]])
        %0 = "d2m.tile_maximum"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        // Check that maximum result is stored back to dst memory space
        // CHECK: affine.store %[[MAXIMUM_RESULT]], %[[DST]]
        // Check that result is loaded from dst memory space
        // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]]
        // Check that final result is stored back to original #l1 memory space
        // CHECK: affine.store %[[FINAL_VAL]], %[[ARG2:.*]]
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @intermediates_thru_dst_chain_2
  func.func @intermediates_thru_dst_chain_2(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                                            %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                                            %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %cb0 = d2m.pop %arg0_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.pop %arg1_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
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
            %0 = "d2m.tile_div"(%in, %in_17) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: %[[DIV_RESULT:.*]] = "d2m.tile_div"(%[[DST0_VAL:.*]], %[[DST1_VAL:.*]]) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: affine.store %[[DIV_RESULT]], %[[DST:.*]][2, %[[ARG_I:.*]], %[[ARG_J:.*]]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: %[[DST_DIV:.*]] = affine.load %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            %1 = "d2m.tile_recip"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: %[[RECIP_RESULT:.*]] = "d2m.tile_recip"(%[[DST_DIV]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: affine.store %[[RECIP_RESULT]], %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: affine.store %[[FINAL_VAL]], {{.*}} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            linalg.yield %1 : !ttcore.tile<32x32, f32>
          }
        }
      }
    }
    return
  }

  // CHECK-LABEL: func.func @intermediates_thru_dst_chain_3
  func.func @intermediates_thru_dst_chain_3(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                                            %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>,
                                            %out0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %cb0 = d2m.pop %arg0_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.pop %arg1_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
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
            // CHECK: %[[SUB_RESULT:.*]] = "d2m.tile_sub"(%[[DST0_VAL:.*]], %[[DST1_VAL:.*]]) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: affine.store %[[SUB_RESULT]], %[[DST:.*]][2, %[[ARG_I:.*]], %[[ARG_J:.*]]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: %[[DST_SUB:.*]] = affine.load %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            %1 = "d2m.tile_eqz"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: %[[EQZ1_RESULT:.*]] = "d2m.tile_eqz"(%[[DST_SUB]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: affine.store %[[EQZ1_RESULT]], %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: %[[DST_EQZ1:.*]] = affine.load %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            %2 = "d2m.tile_eqz"(%1) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: %[[EQZ2_RESULT:.*]] = "d2m.tile_eqz"(%[[DST_EQZ1]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: affine.store %[[EQZ2_RESULT]], %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]][2, %[[ARG_I]], %[[ARG_J]]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: affine.store %[[FINAL_VAL]], {{.*}} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            linalg.yield %2 : !ttcore.tile<32x32, f32>
          }
        }
      }
    }
    return
  }

  // CHECK-LABEL: func.func @eltwise_unary_chain_multi_tile
  func.func @eltwise_unary_chain_multi_tile(%in0: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>,
                                            %out0: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>)
        outs(%out0 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>, %arg1_cb: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>):
      %cb0 = d2m.pop %arg0_cb : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.for %arg1 = %c0 to %c4 step %c2 {
        scf.for %arg2 = %c0 to %c4 step %c4 {
          %subview = memref.subview %cb0[%arg1, %arg2] [2, 4] [1, 1] : memref<4x4x!ttcore.tile<32x32, f32>, #l1_> to memref<2x4x!ttcore.tile<32x32, f32>, strided<[4, 1], offset: ?>, #l1_>
          %subview_4 = memref.subview %cb1[%arg1, %arg2] [2, 4] [1, 1] : memref<4x4x!ttcore.tile<32x32, f32>, #l1_> to memref<2x4x!ttcore.tile<32x32, f32>, strided<[4, 1], offset: ?>, #l1_>
          %dst = d2m.acquire_dst() : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 4 {
              %0 = affine.load %subview[%arg3, %arg4] : memref<2x4x!ttcore.tile<32x32, f32>, strided<[4, 1], offset: ?>, #l1_>
              affine.store %0, %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
            }
          }
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 4 {
              %0 = affine.load %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
              %1 = "d2m.tile_abs"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              // CHECK: %[[ABS_RESULT:.*]] = "d2m.tile_abs"(%[[DST0_VAL:.*]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              // CHECK: affine.store %[[ABS_RESULT]], %dst[0, %[[ARG_I:.*]], %[[ARG_J:.*]]] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst>
              affine.store %1, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
              %2 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
              // CHECK: %[[DST_ABS:.*]] = affine.load %dst[0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst>
              %3 = "d2m.tile_sin"(%2) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              // CHECK: %[[SIN_RESULT:.*]] = "d2m.tile_sin"(%[[DST_ABS]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              // CHECK: affine.store %[[SIN_RESULT]], %dst[0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst>
              affine.store %3, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
              %4 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
              // CHECK: %[[DST_SIN:.*]] = affine.load %dst[0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst>
              %5 = "d2m.tile_negative"(%4) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              // CHECK: %[[NEG_RESULT:.*]] = "d2m.tile_negative"(%[[DST_SIN]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              // CHECK: affine.store %[[NEG_RESULT]], %dst[0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst>
              affine.store %5, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
              %6 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
              // CHECK: %[[DST_NEG:.*]] = affine.load %dst[0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst>
              %7 = "d2m.tile_exp"(%6) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              // CHECK: %[[EXP_RESULT:.*]] = "d2m.tile_exp"(%[[DST_NEG]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              // CHECK: affine.store %[[EXP_RESULT]], %dst[0, %[[ARG_I]], %[[ARG_J]]] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst>
              affine.store %7, %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
            }
          }
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 4 {
              %0 = affine.load %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst_>
              // CHECK: %[[FINAL_VAL:.*]] = affine.load %dst[0, %[[ARG_I:.*]], %[[ARG_J:.*]]] : memref<1x2x4x!ttcore.tile<32x32, f32>, #dst>
              // CHECK: affine.store %[[FINAL_VAL]], {{.*}} : memref<2x4x!ttcore.tile<32x32, f32>, strided<[4, 1], offset: ?>, #l1>
              affine.store %0, %subview_4[%arg3, %arg4] : memref<2x4x!ttcore.tile<32x32, f32>, strided<[4, 1], offset: ?>, #l1_>
            }
          }
        }
      }
    }
    return
  }
}
