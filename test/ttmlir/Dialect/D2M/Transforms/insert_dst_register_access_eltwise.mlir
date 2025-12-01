// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-access='allocation-strategy=basic' --canonicalize %s | FileCheck %s --check-prefixes=CHECK,BASIC
// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-access='allocation-strategy=greedy' --canonicalize %s | FileCheck %s --check-prefixes=CHECK,GREEDY
// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-access='allocation-strategy=chaitin' --canonicalize %s | FileCheck %s --check-prefixes=CHECK,CHAITIN
#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @binary
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: %[[ARG0_VAL:.*]] = affine.load %[[ARG0:.*]]
  // CHECK: affine.store %[[ARG0_VAL]], %[[DST]]
  // CHECK: %[[ARG1_VAL:.*]] = affine.load %[[ARG1:.*]]
  // CHECK: affine.store %[[ARG1_VAL]], %[[DST]]
  // CHECK: %[[DST0_VAL:.*]] = affine.load %[[DST]]
  // CHECK: %[[DST1_VAL:.*]] = affine.load %[[DST]]
  // CHECK: %[[MAXIMUM_RESULT:.*]] = "d2m.tile_maximum"(%[[DST0_VAL]], %[[DST1_VAL]])
  // CHECK: affine.store %[[MAXIMUM_RESULT]], %[[DST]]
  // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]]
  // CHECK: affine.store %[[FINAL_VAL]], %[[ARG2:.*]]
  func.func @binary(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
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
      %subview = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_1 = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_2 = memref.subview %cb2[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_maximum"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
  }
    return
  }

  // CHECK-LABEL: func.func @binary_comparison_chain
  // CHECK: %[[DST:.*]] = d2m.acquire_dst()
  // CHECK: affine.for %[[IDX_I1:.*]] =
  // CHECK: affine.for %[[IDX_J1:.*]] =
  // CHECK: %[[ARG0_VAL:.*]] = affine.load %[[ARG0:.*]]
  // CHECK: affine.store %[[ARG0_VAL]], %[[DST]][0, %[[IDX_I1]], %[[IDX_J1]]]
  // CHECK: %[[ARG1_VAL:.*]] = affine.load %[[ARG1:.*]]
  // CHECK: affine.store %[[ARG1_VAL]], %[[DST]][1, %[[IDX_I1]], %[[IDX_J1]]]

  // CHECK: affine.for %[[IDX_I2:.*]] =
  // CHECK: affine.for %[[IDX_J2:.*]] =
  // CHECK: %[[DST0:.*]] = affine.load %[[DST]][0, %[[IDX_I2]], %[[IDX_J2]]]
  // CHECK: %[[DST1:.*]] = affine.load %[[DST]][1, %[[IDX_I2]], %[[IDX_J2]]]
  // CHECK: %[[SUB_RESULT:.*]] = "d2m.tile_sub"(%[[DST0]], %[[DST1]])
  // CHECK: affine.store %[[SUB_RESULT]], %[[DST]][2, %arg3, %arg4] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: %[[SUB_LOADED:.*]] = affine.load %[[DST]][2, %arg3, %arg4] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: %[[NEZ_RESULT:.*]] = "d2m.tile_nez"(%[[SUB_LOADED]])
  // CHECK: affine.store %[[NEZ_RESULT]], %[[DST]][2, %arg3, %arg4] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst>

  // CHECK: affine.for %[[IDX_I3:.*]] =
  // CHECK: affine.for %[[IDX_J3:.*]] =
  // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]][2, %arg3, %arg4] : memref<4x1x1x!ttcore.tile<32x32, f32>, #dst>
  // CHECK: affine.store %[[FINAL_VAL]], %[[ARG2:.*]]
  func.func @binary_comparison_chain(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
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
      %subview = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_1 = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_2 = memref.subview %cb2[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        %1 = "d2m.tile_nez"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %1 : !ttcore.tile<32x32, f32>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @intermediates_thru_dst_chain_2
  // CHECK: %[[CB_OUT:.*]] = d2m.reserve %[[CB_SRC:[^ ]+]] : <memref<1x1x!ttcore.tile<32x32, f16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1>
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: affine.for %[[ARG_I1:.*]] =
  // CHECK: affine.for %[[ARG_J1:.*]] =
  // CHECK: %[[ARG0_VAL:.*]] = affine.load %[[ARG0:.*]][%[[ARG_I1]], %[[ARG_J1]]] : memref<1x1x!ttcore.tile<32x32, f16>, #l1>
  // BASIC: affine.store %[[ARG0_VAL]], %[[DST]][0, %[[ARG_I1]], %[[ARG_J1]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // GREEDY: affine.store %[[ARG0_VAL]], %[[DST]][0, %[[ARG_I1]], %[[ARG_J1]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHAITIN: affine.store %[[ARG0_VAL]], %[[DST]][0, %[[ARG_I1]], %[[ARG_J1]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[ARG1_VAL:.*]] = affine.load %[[ARG1:.*]][%[[ARG_I1]], %[[ARG_J1]]] : memref<1x1x!ttcore.tile<32x32, f16>, #l1>
  // CHECK: affine.store %[[ARG1_VAL]], %[[DST]][1, %[[ARG_I1]], %[[ARG_J1]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>

  // CHECK: affine.for %[[ARG_I2:.*]] =
  // CHECK: affine.for %[[ARG_J2:.*]] =
  // CHECK: %[[DST0_VAL:.*]] = affine.load %[[DST]][0, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[DST1_VAL:.*]] = affine.load %[[DST]][1, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[DIV_RESULT:.*]] = "d2m.tile_div"(%[[DST0_VAL]], %[[DST1_VAL]]) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
  // CHECK: affine.store %[[DIV_RESULT]], %[[DST]][2, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[DST_DIV:.*]] = affine.load %[[DST]][2, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[RECIP_RESULT:.*]] = "d2m.tile_recip"(%[[DST_DIV]]) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
  // CHECK: affine.store %[[RECIP_RESULT]], %[[DST]][2, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>

  // CHECK: affine.for %[[ARG_I3:.*]] =
  // CHECK: affine.for %[[ARG_J3:.*]] =
  // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]][2, %[[ARG_I3]], %[[ARG_J3]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: affine.store %[[FINAL_VAL]], %[[CB_OUT]][%[[ARG_I3]], %[[ARG_J3]]] : memref<1x1x!ttcore.tile<32x32, f16>, #l1>
  func.func @intermediates_thru_dst_chain_2(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>,
                                            %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>,
                                            %out0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
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
            %1 = "d2m.tile_recip"(%0) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
            linalg.yield %1 : !ttcore.tile<32x32, f16>
          }
        }
      }
    }
    return
  }

  // CHECK-LABEL: func.func @intermediates_thru_dst_chain_3
  // CHECK: %[[CBOUT_3:.*]] = d2m.reserve %[[CB_SRC3:[^ ]+]] : <memref<1x1x!ttcore.tile<32x32, f16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1>
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: affine.for %[[ARG_I1:.*]] =
  // CHECK: affine.for %[[ARG_J1:.*]] =
  // CHECK: %[[ARG0_VAL:.*]] = affine.load %[[ARG0:.*]][%[[ARG_I1]], %[[ARG_J1]]] : memref<1x1x!ttcore.tile<32x32, f16>, #l1>
  // CHECK: affine.store %[[ARG0_VAL]], %[[DST]][0, %[[ARG_I1]], %[[ARG_J1]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[ARG1_VAL:.*]] = affine.load %[[ARG1:.*]][%[[ARG_I1]], %[[ARG_J1]]] : memref<1x1x!ttcore.tile<32x32, f16>, #l1>
  // CHECK: affine.store %[[ARG1_VAL]], %[[DST]][1, %[[ARG_I1]], %[[ARG_J1]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>


  // CHECK: affine.for %[[ARG_I2:.*]] =
  // CHECK: affine.for %[[ARG_J2:.*]] =
  // CHECK: %[[DST0:.*]] = affine.load %[[DST]][0, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[DST1:.*]] = affine.load %[[DST]][1, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[SUB_RESULT:.*]] = "d2m.tile_sub"(%[[DST0]], %[[DST1]]) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
  // CHECK: affine.store %[[SUB_RESULT]], %[[DST]][2, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[DST_SUB:.*]] = affine.load %[[DST]][2, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[EQZ1_RESULT:.*]] = "d2m.tile_eqz"(%[[DST_SUB]]) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
  // BASIC: affine.store %[[EQZ1_RESULT]], %[[DST]][2, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // GREEDY: affine.store %[[EQZ1_RESULT]], %[[DST]][0, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHAITIN: affine.store %[[EQZ1_RESULT]], %[[DST]][0, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // BASIC: %[[DST_EQZ1:.*]] = affine.load %[[DST]][2, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // GREEDY: %[[DST_EQZ1:.*]] = affine.load %[[DST]][0, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHAITIN: %[[DST_EQZ1:.*]] = affine.load %[[DST]][0, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: %[[EQZ2_RESULT:.*]] = "d2m.tile_eqz"(%[[DST_EQZ1]]) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
  // BASIC: affine.store %[[EQZ2_RESULT]], %[[DST]][2, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // GREEDY: affine.store %[[EQZ2_RESULT]], %[[DST]][0, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHAITIN: affine.store %[[EQZ2_RESULT]], %[[DST]][0, %[[ARG_I2]], %[[ARG_J2]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>

  // CHECK: affine.for %[[ARG_I3:.*]] =
  // CHECK: affine.for %[[ARG_J3:.*]] =
  // BASIC: %[[FINAL_VAL:.*]] = affine.load %[[DST]][2, %[[ARG_I3]], %[[ARG_J3]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // GREEDY: %[[FINAL_VAL:.*]] = affine.load %[[DST]][0, %[[ARG_I3]], %[[ARG_J3]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHAITIN: %[[FINAL_VAL:.*]] = affine.load %[[DST]][0, %[[ARG_I3]], %[[ARG_J3]]] : memref<8x1x1x!ttcore.tile<32x32, f16>, #dst>
  // CHECK: affine.store %[[FINAL_VAL]], %[[CBOUT_3]][%[[ARG_I3]], %[[ARG_J3]]] : memref<1x1x!ttcore.tile<32x32, f16>, #l1>
  func.func @intermediates_thru_dst_chain_3(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>,
                                            %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>,
                                            %out0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<2048x2048, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1 {
        scf.for %arg1 = %c0 to %c1 step %c1 {
          %subview = memref.subview %cb0[%arg0, %arg1] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
          %subview_1 = memref.subview %cb1[%arg0, %arg1] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
          %subview_2 = memref.subview %cb2[%arg0, %arg1] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_> to memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f16>, strided<[1, 1], offset: ?>, #l1_>) {
          ^bb0(%in: !ttcore.tile<32x32, f16>, %in_17: !ttcore.tile<32x32, f16>, %out: !ttcore.tile<32x32, f16>):
            %0 = "d2m.tile_sub"(%in, %in_17) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
            %1 = "d2m.tile_eqz"(%0) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
            %2 = "d2m.tile_eqz"(%1) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
            linalg.yield %2 : !ttcore.tile<32x32, f16>
          }
        }
      }
    }
    return
  }

  // CHECK-LABEL: func.func @eltwise_unary_chain_multi_tile
  // CHECK: scf.for %[[ARG_OUTER:.*]] =
  // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
  // CHECK: affine.for %[[ARG_I1:.*]] = 0 to 2 {
  // CHECK: affine.for %[[ARG_J1:.*]] = 0 to 4 {
  // CHECK: %[[INPUT_VAL:.*]] = affine.load %subview[%arg3, %arg4] : memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #l1>
  // CHECK: affine.store %[[INPUT_VAL]], %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>

  // CHECK: affine.for %[[ARG_I2:.*]] = 0 to 2 {
  // CHECK: affine.for %[[ARG_J2:.*]] = 0 to 4 {
  // CHECK: %[[DST0_VAL:.*]] = affine.load %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
  // CHECK: %[[ABS_RESULT:.*]] = "d2m.tile_abs"(%[[DST0_VAL]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
  // CHECK: affine.store %[[ABS_RESULT]], %dst[0, %[[ARG_I2]], %[[ARG_J2]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
  // CHECK: %[[DST_ABS:.*]] = affine.load %dst[0, %[[ARG_I2]], %[[ARG_J2]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
  // CHECK: %[[SIN_RESULT:.*]] = "d2m.tile_sin"(%[[DST_ABS]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
  // CHECK: affine.store %[[SIN_RESULT]], %dst[0, %[[ARG_I2]], %[[ARG_J2]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
  // CHECK: %[[DST_SIN:.*]] = affine.load %dst[0, %[[ARG_I2]], %[[ARG_J2]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
  // CHECK: %[[NEG_RESULT:.*]] = "d2m.tile_negative"(%[[DST_SIN]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
  // CHECK: affine.store %[[NEG_RESULT]], %dst[0, %[[ARG_I2]], %[[ARG_J2]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
  // CHECK: %[[DST_NEG:.*]] = affine.load %dst[0, %[[ARG_I2]], %[[ARG_J2]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
  // CHECK: %[[EXP_RESULT:.*]] = "d2m.tile_exp"(%[[DST_NEG]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
  // CHECK: affine.store %[[EXP_RESULT]], %dst[0, %[[ARG_I2]], %[[ARG_J2]]] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>

  // CHECK: affine.for %[[ARG_I3:.*]] = 0 to 2 {
  // CHECK: affine.for %[[ARG_J3:.*]] = 0 to 4 {
  // CHECK: %[[FINAL_VAL:.*]] = affine.load %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
  // CHECK: affine.store %[[FINAL_VAL]], %subview_0[%arg3, %arg4] : memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #l1>
  func.func @eltwise_unary_chain_multi_tile(%in0: memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048, 1>, #l1_>,
                                            %out0: memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>>, %arg1_cb: !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, bf16>, #l1_>
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
              affine.store %1, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              %2 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              %3 = "d2m.tile_sin"(%2) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              affine.store %3, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              %4 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              %5 = "d2m.tile_negative"(%4) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              affine.store %5, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              %6 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              %7 = "d2m.tile_exp"(%6) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              affine.store %7, %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
            }
          }
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 4 {
              %0 = affine.load %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
              affine.store %0, %subview_4[%arg3, %arg4] : memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #l1_>
            }
          }
        }
      }
    }
    return
  }

  // Composite function test: two ops produce results consumed by a third. Here,
  // two parallel tile_recip operations feed tile_add - they must use different
  // DST slots since tile_add needs both values simultaneously.

  // CHECK-LABEL: func.func @two_recip_then_add
  // CHECK: %[[DST:.*]] = d2m.acquire_dst()
  // CHECK: "d2m.tile_recip"
  // BASIC-NEXT: affine.store %{{.*}}, %[[DST]][0,
  // GREEDY-NEXT: affine.store %{{.*}}, %[[DST]][0,
  // CHAITIN-NEXT: affine.store %{{.*}}, %[[DST]][0,
  // CHECK: "d2m.tile_recip"
  // BASIC-NEXT: affine.store %{{.*}}, %[[DST]][1,
  // GREEDY-NEXT: affine.store %{{.*}}, %[[DST]][1,
  // CHAITIN-NEXT: affine.store %{{.*}}, %[[DST]][1,

  func.func @two_recip_then_add(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
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
      %subview = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_1 = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      %subview_2 = memref.subview %cb2[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_recip"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        %1 = "d2m.tile_recip"(%arg1) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        %2 = "d2m.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %2 : !ttcore.tile<32x32, f32>
      }
  }
    return
  }

  // This test is checking that we never produce a negative DST slice index
  // CHECK-LABEL: func.func @basic_unary_chain_alloc
  // CHECK: d2m.generic
  // CHECK: d2m.acquire_dst
  // CHECK-NOT: %dst[-1
  func.func @basic_unary_chain_alloc(
      %in0: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x4096, 1>, #l1_>,
      %out0: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
                 indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                  affine_map<(d0, d1) -> (d0, d1)>],
                 iterator_types = [#ttcore.iterator_type<parallel>,
                                   #ttcore.iterator_type<parallel>],
                 threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>>,
              %arg1_cb: !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>
      %c0 = arith.constant 0 : index
      %subview = memref.subview %cb0[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1_> to memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1_>
      %subview_out = memref.subview %cb1[%c0, %c0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1_> to memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1_>
      linalg.generic
          {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>],
           iterator_types = ["parallel", "parallel"]}
          ins(%subview : memref<1x1x!ttcore.tile<32x32, bf16>,
                                strided<[1, 1], offset: ?>, #l1_>)
          outs(%subview_out : memref<1x1x!ttcore.tile<32x32, bf16>,
                                     strided<[1, 1], offset: ?>, #l1_>) {
      ^bb0(%arg0: !ttcore.tile<32x32, bf16>, %arg1: !ttcore.tile<32x32, bf16>):
        %0 = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        %1 = "d2m.tile_sin"(%0) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        %2 = "d2m.tile_negative"(%1) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        %3 = "d2m.tile_exp"(%2) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        linalg.yield %3 : !ttcore.tile<32x32, bf16>
      }
    }
    return
  }
}
