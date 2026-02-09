// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-access --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  // CHECK-LABEL: func.func @bcast_col
  func.func @bcast_col(%in0: memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                       %in1: memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                       %out0: memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    %alloc = memref.alloc() {address = 102208 : i64, alignment = 16 : i64} : memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>
    %stream = "d2m.stream_layout"(%in1, %alloc) <{remapping = #map4}> : (memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>) -> memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<3x3>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%in0, %stream : memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_>)
        outs(%out0 : memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^unified0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %0 = d2m.wait %cb0 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %1 = d2m.wait %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %2 = d2m.reserve %cb2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1_9 = arith.constant 1 : index
      %c1_10 = arith.constant 1 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1_9 step %c1_10 {
        scf.for %arg3 = %c0_11 to %c1_12 step %c1_13 {
          %subview = memref.subview %0[%arg2, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview_1 = memref.subview %1[%arg2, 0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview_2 = memref.subview %2[%arg2, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          // CHECK: %[[DIM1:.*]] = d2m.iter_index(1) : index
          // CHECK-NEXT: %[[GUARD:.*]] = arith.cmpi eq, %[[DIM1]], %{{.*}} : index
          // CHECK-NEXT: scf.if %[[GUARD]]
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: %[[L1_TILE:.*]] = affine.load
          // CHECK-NEXT: %[[DST_TILE:.*]] = "d2m.tile_bcast"(%[[L1_TILE]]) <{bcast_type = #d2m<tile_bcast_type col>}>
          // CHECK-NEXT: affine.store %[[DST_TILE]], %dst
          // CHECK: affine.for
          // CHECK: affine.for
          // CHECK: affine.load %dst
          // CHECK: affine.load %dst
          // CHECK: d2m.tile_div
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %in_1: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_bcast"(%in_1) <{bcast_type = #d2m<tile_bcast_type col>}> : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            %4 = "d2m.tile_div"(%in, %3) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %4 : !ttcore.tile<32x32, f32>
          }
        }
      }
    }
    memref.dealloc %alloc : memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @bcast_row
  func.func @bcast_row(%in0: memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                       %in1: memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                       %out0: memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    %alloc = memref.alloc() {address = 102208 : i64, alignment = 16 : i64} : memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>
    %stream = "d2m.stream_layout"(%in1, %alloc) <{remapping = #map4}> : (memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>) -> memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<3x3>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%in0, %stream : memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_>)
        outs(%out0 : memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^unified0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %0 = d2m.wait %cb0 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %1 = d2m.wait %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %2 = d2m.reserve %cb2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1_9 = arith.constant 1 : index
      %c1_10 = arith.constant 1 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1_9 step %c1_10 {
        scf.for %arg3 = %c0_11 to %c1_12 step %c1_13 {
          %subview = memref.subview %0[%arg2, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview_1 = memref.subview %1[0, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview_2 = memref.subview %2[%arg2, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          // CHECK: %[[DIM0:.*]] = d2m.iter_index(0) : index
          // CHECK-NEXT: %[[GUARD:.*]] = arith.cmpi eq, %[[DIM0]], %{{.*}} : index
          // CHECK-NEXT: scf.if %[[GUARD]]
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: %[[L1_TILE:.*]] = affine.load
          // CHECK-NEXT: %[[DST_TILE:.*]] = "d2m.tile_bcast"(%[[L1_TILE]]) <{bcast_type = #d2m<tile_bcast_type row>}>
          // CHECK-NEXT: affine.store %[[DST_TILE]], %dst
          // CHECK: affine.for
          // CHECK: affine.for
          // CHECK: affine.load %dst
          // CHECK: affine.load %dst
          // CHECK: d2m.tile_div
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %in_1: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_bcast"(%in_1) <{bcast_type = #d2m<tile_bcast_type row>}> : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            %4 = "d2m.tile_div"(%in, %3) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %4 : !ttcore.tile<32x32, f32>
          }
        }
      }
    }
    memref.dealloc %alloc : memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @bcast_scalar
  func.func @bcast_scalar(%in0: memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                          %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                          %out0: memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    %alloc = memref.alloc() {address = 102208 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>
    %stream = "d2m.stream_layout"(%in1, %alloc) <{remapping = #map4}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>) -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<3x3>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%in0, %stream : memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_>)
        outs(%out0 : memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^unified0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %0 = d2m.wait %cb0 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %1 = d2m.wait %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %2 = d2m.reserve %cb2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1_9 = arith.constant 1 : index
      %c1_10 = arith.constant 1 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1_9 step %c1_10 {
        scf.for %arg3 = %c0_11 to %c1_12 step %c1_13 {
          %subview = memref.subview %0[%arg2, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview_1 = memref.subview %2[%arg2, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          // CHECK: %[[DIM0:.*]] = d2m.iter_index(0) : index
          // CHECK-NEXT: %[[GUARD0:.*]] = arith.cmpi eq, %[[DIM0]], %{{.*}} : index
          // CHECK: %[[DIM1:.*]] = d2m.iter_index(1) : index
          // CHECK-NEXT: %[[GUARD1:.*]] = arith.cmpi eq, %[[DIM1]], %{{.*}} : index
          // CHECK-NEXT: %[[GUARD:.*]] = arith.andi %[[GUARD0]], %[[GUARD1]] : i1
          // CHECK-NEXT: scf.if %[[GUARD]]
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: %[[L1_TILE:.*]] = affine.load
          // CHECK-NEXT: %[[DST_TILE:.*]] = "d2m.tile_bcast"(%[[L1_TILE]]) <{bcast_type = #d2m<tile_bcast_type scalar>}>
          // CHECK-NEXT: affine.store %[[DST_TILE]], %dst
          // CHECK: affine.for
          // CHECK: affine.for
          // CHECK: affine.load %dst
          // CHECK: affine.load %dst
          // CHECK: d2m.tile_div
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_>) outs(%subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %in_1: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_bcast"(%in_1) <{bcast_type = #d2m<tile_bcast_type scalar>}> : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            %4 = "d2m.tile_div"(%in, %3) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %4 : !ttcore.tile<32x32, f32>
          }
        }
      }
    }
    memref.dealloc %alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @bcast_dual
  func.func @bcast_dual(%in0: memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                        %in1: memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                        %out0: memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
    %alloc = memref.alloc() {address = 102208 : i64, alignment = 16 : i64} : memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>
    %stream = "d2m.stream_layout"(%in0, %alloc) <{remapping = #map4}> : (memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>) -> memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_>
    %alloc_1 = memref.alloc() {address = 110400 : i64, alignment = 16 : i64} : memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>
    %stream_1 = "d2m.stream_layout"(%in1, %alloc_1) <{remapping = #map4}> : (memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>) -> memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<3x3>, indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%stream, %stream_1 : memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_>, memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1_>)
        outs(%out0 : memref<3x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^unified0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
      %0 = d2m.wait %cb0 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %1 = d2m.wait %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %2 = d2m.reserve %cb2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      %c0_14 = arith.constant 0 : index
      %c1_15 = arith.constant 1 : index
      %c1_16 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1_12 step %c1_13 {
        scf.for %arg3 = %c0_14 to %c1_15 step %c1_16 {
          %subview = memref.subview %0[%arg2, 0] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview_1 = memref.subview %1[0, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          %subview_2 = memref.subview %2[%arg2, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>
          // CHECK: %[[DIM1:.*]] = d2m.iter_index(1) : index
          // CHECK-NEXT: %[[GUARD1:.*]] = arith.cmpi eq, %[[DIM1]], %{{.*}} : index
          // CHECK-NEXT: scf.if %[[GUARD1]]
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: %[[L1_TILE1:.*]] = affine.load
          // CHECK-NEXT: %[[DST_TILE1:.*]] = "d2m.tile_bcast"(%[[L1_TILE1]]) <{bcast_type = #d2m<tile_bcast_type col>}>
          // CHECK-NEXT: affine.store %[[DST_TILE1]], %dst

          // CHECK: %[[DIM0:.*]] = d2m.iter_index(0) : index
          // CHECK-NEXT: %[[GUARD0:.*]] = arith.cmpi eq, %[[DIM0]], %{{.*}} : index
          // CHECK-NEXT: scf.if %[[GUARD0]]
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: affine.for
          // CHECK-NEXT: %[[L1_TILE0:.*]] = affine.load
          // CHECK-NEXT: %[[DST_TILE0:.*]] = "d2m.tile_bcast"(%[[L1_TILE0]]) <{bcast_type = #d2m<tile_bcast_type row>}>
          // CHECK-NEXT: affine.store %[[DST_TILE0]], %dst

          // CHECK: affine.for
          // CHECK: affine.for
          // CHECK: affine.load %dst
          // CHECK: affine.load %dst
          // CHECK: d2m.tile_div
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_1 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) outs(%subview_2 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #l1_>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %in_1: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_bcast"(%in) <{bcast_type = #d2m<tile_bcast_type col>}> : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            %4 = "d2m.tile_bcast"(%in_1) <{bcast_type = #d2m<tile_bcast_type row>}> : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            %5 = "d2m.tile_div"(%3, %4) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %5 : !ttcore.tile<32x32, f32>
          }
        }
      }
    }
    memref.dealloc %alloc : memref<3x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>
    memref.dealloc %alloc_1 : memref<1x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 2>, #l1_>
    return
  }
}
