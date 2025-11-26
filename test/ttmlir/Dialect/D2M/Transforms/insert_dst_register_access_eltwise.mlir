// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-access --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefixes=CHECK,LEGACY
// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-gc="coloring-strategy=greedy" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,GC
// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-gc="coloring-strategy=chaitin-briggs" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,GC

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @binary
  // CHECK: d2m.acquire_dst
  // CHECK: affine.for
  // LEGACY: affine.load %{{.*}}[0,
  // LEGACY: affine.load %{{.*}}[1,
  // CHECK: d2m.tile_maximum
  // CHECK-SAME: result_dst_index =
  // CHECK: d2m.release_dst
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

  // In the legacy pass, every op gets result_dst_index and stores to DST.
  // In the GC pass, intermediate ops don't get result_dst_index (values stay in registers).
  // CHECK-LABEL: func.func @intermediates_thru_dst_chain_2
  // CHECK: d2m.acquire_dst
  // CHECK: affine.for
  // CHECK: d2m.tile_div
  // LEGACY-SAME: result_dst_index = 2
  // GC-NOT: result_dst_index
  // CHECK: d2m.tile_recip
  // LEGACY-SAME: result_dst_index = 2
  // GC-SAME: result_dst_index = 2
  // CHECK: d2m.release_dst
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

  // Same pattern: only final op (last tile_eqz) gets result_dst_index in GC pass.
  // CHECK-LABEL: func.func @intermediates_thru_dst_chain_3
  // CHECK: d2m.acquire_dst
  // CHECK: affine.for
  // CHECK: d2m.tile_sub
  // LEGACY-SAME: result_dst_index = 2
  // GC-NOT: result_dst_index
  // CHECK: d2m.tile_eqz
  // LEGACY-SAME: result_dst_index = 2
  // GC-NOT: result_dst_index
  // CHECK: d2m.tile_eqz
  // LEGACY-SAME: result_dst_index = 2g
  // GC-SAME: result_dst_index = 2
  // CHECK: d2m.release_dst
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

  // This function already has explicit DST handling (acquire/release inside the scf.for loop).
  // Both passes skip this function since it's already allocated.
  // CHECK-LABEL: func.func @eltwise_unary_chain_multi_tile
  // CHECK: d2m.acquire_dst
  // CHECK: affine.for
  // CHECK: d2m.tile_abs
  // CHECK: d2m.tile_sin
  // CHECK: d2m.tile_negative
  // CHECK: d2m.tile_exp
  // CHECK: d2m.release_dst
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
          d2m.release_dst %dst : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst_>
        }
      }
    }
    return
  }
}
