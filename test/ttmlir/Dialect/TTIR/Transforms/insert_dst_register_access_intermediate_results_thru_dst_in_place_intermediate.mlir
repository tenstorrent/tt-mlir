// RUN: ttmlir-opt --ttcore-register-device --ttir-insert-dst-register-access="use-tile-matmul=false" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
module {
  func.func @eq(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) -> memref<4x4xf32> {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %alloc = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
    %alloc_1 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    %alloc_2 = memref.alloc() : memref<4x4xf32, #ttcore.host_layout<logical_shape = 4x4, host_strides = 32x1, host_volume = 1024>>
    memref.copy %arg0, %alloc_2 : memref<4x4xf32> to memref<4x4xf32, #ttcore.host_layout<logical_shape = 4x4, host_strides = 32x1, host_volume = 1024>>
    ttir.to_layout %alloc_2, %alloc_1 : memref<4x4xf32, #ttcore.host_layout<logical_shape = 4x4, host_strides = 32x1, host_volume = 1024>> into memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>> hostInfo = <logical_shape = 4x4, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
    ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#ttir.thread<compute>]}
        ins(%alloc_1 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>)
        outs(%alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)  {
    ^compute0(%cb0: memref<32x32xf32, #ttcore.memory_space<l1>>, %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>):
      "ttir.tile_tilize_block"(%cb0, %cb1) : (memref<32x32xf32, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>) -> ()
    }
    memref.dealloc %alloc_1 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    %alloc_3 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
    %alloc_4 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    %alloc_5 = memref.alloc() : memref<4x4xf32, #ttcore.host_layout<logical_shape = 4x4, host_strides = 32x1, host_volume = 1024>>
    memref.copy %arg1, %alloc_5 : memref<4x4xf32> to memref<4x4xf32, #ttcore.host_layout<logical_shape = 4x4, host_strides = 32x1, host_volume = 1024>>
    ttir.to_layout %alloc_5, %alloc_4 : memref<4x4xf32, #ttcore.host_layout<logical_shape = 4x4, host_strides = 32x1, host_volume = 1024>> into memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>> hostInfo = <logical_shape = 4x4, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
    ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#ttir.thread<compute>]}
        ins(%alloc_4 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>)
        outs(%alloc_3 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)  {
    ^compute0(%cb0: memref<32x32xf32, #ttcore.memory_space<l1>>, %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>):
      "ttir.tile_tilize_block"(%cb0, %cb1) : (memref<32x32xf32, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>) -> ()
    }
    memref.dealloc %alloc_4 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    %alloc_6 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
    ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#ttir.thread<compute>]}
        ins(%alloc, %alloc_3 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)
        outs(%alloc_6 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)  {
    ^compute0(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %cb2: memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>):
      %c0 = arith.constant 0 : index
      %c1_10 = arith.constant 1 : index
      %c1_11 = arith.constant 1 : index
      %c0_12 = arith.constant 0 : index
      %c1_13 = arith.constant 1 : index
      %c1_14 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1_10 step %c1_11 {
        scf.for %arg3 = %c0_12 to %c1_13 step %c1_14 {
          %subview = memref.subview %cb0[%arg2, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_15 = memref.subview %cb1[%arg2, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_16 = memref.subview %cb2[%arg2, %arg3] [1, 1] [1, 1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_15 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>) outs(%subview_16 : memref<1x1x!ttcore.tile<32x32, f32>, strided<[1, 1], offset: ?>, #ttcore.memory_space<l1>>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %in_17: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %0 = "ttir.tile_sub_binary"(%in, %in_17) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: %[[SUB_RESULT:.*]] = "ttir.tile_sub_binary"(%[[DST0_VAL:.*]], %[[DST1_VAL:.*]]) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: affine.store %[[SUB_RESULT]], %[[DST:.*]][2, %arg2, %arg3] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: %[[DST_SUB:.*]] = affine.load %[[DST]][2, %arg2, %arg3] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            %1 = "ttir.tile_eqz"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: %[[EQZ1_RESULT:.*]] = "ttir.tile_eqz"(%[[DST_SUB]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: affine.store %[[EQZ1_RESULT]], %[[DST]][2, %arg2, %arg3] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: %[[DST_EQZ1:.*]] = affine.load %[[DST]][2, %arg2, %arg3] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            %2 = "ttir.tile_eqz"(%1) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: %[[EQZ2_RESULT:.*]] = "ttir.tile_eqz"(%[[DST_EQZ1]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            // CHECK: affine.store %[[EQZ2_RESULT]], %[[DST]][2, %arg2, %arg3] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: %[[FINAL_VAL:.*]] = affine.load %[[DST]][2, %arg2, %arg3] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst>
            // CHECK: affine.store %[[FINAL_VAL]], %cb2[%arg2, %arg3] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            linalg.yield %2 : !ttcore.tile<32x32, f32>
          }
        }
      }
    }
    memref.dealloc %alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
    memref.dealloc %alloc_3 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
    %alloc_7 = memref.alloc() : memref<4x4xf32>
    %alloc_8 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#ttir.thread<compute>]}
        ins(%alloc_6 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)
        outs(%alloc_8 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>)  {
    ^compute0(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %cb1: memref<32x32xf32, #ttcore.memory_space<l1>>):
      "ttir.tile_untilize_block"(%cb0, %cb1) : (memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<32x32xf32, #ttcore.memory_space<l1>>) -> ()
    }
    memref.dealloc %alloc_6 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
    %alloc_9 = memref.alloc() : memref<4x4xf32, #ttcore.host_layout<logical_shape = 4x4, host_strides = 32x1, host_volume = 1024>>
    ttir.to_layout %alloc_8, %alloc_9 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>> into memref<4x4xf32, #ttcore.host_layout<logical_shape = 4x4, host_strides = 32x1, host_volume = 1024>> hostInfo = <logical_shape = 4x4, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
    memref.dealloc %alloc_8 : memref<1x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    memref.copy %alloc_9, %alloc_7 : memref<4x4xf32, #ttcore.host_layout<logical_shape = 4x4, host_strides = 32x1, host_volume = 1024>> to memref<4x4xf32>
    return %alloc_7 : memref<4x4xf32>
  }
}
