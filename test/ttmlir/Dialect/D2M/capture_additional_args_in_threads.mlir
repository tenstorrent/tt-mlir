// RUN: ttmlir-opt --split-input-file --capture-additional-args-in-threads %s | FileCheck %s
// Test for capturing additional arguments into threads

#l1 = #ttcore.memory_space<l1>
module {
  func.func @test(%arg0: ui32) {
    %alloc = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_0 : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>)
        outs(%alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        additionalArgs(%arg0 : ui32)
     {
    // CHECK: ^unified0(%cb0: !d2m.cb<memref<32x32xf32, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %scalar0: !d2m.scalar<ui32>):
    ^unified0(%cb0: !d2m.cb<memref<32x32xf32, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
      %0 = d2m.reserve %cb0 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
      d2m.push %cb0 : <memref<32x32xf32, #l1>>
      %1 = d2m.wait %cb0 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
      %2 = d2m.reserve %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %3 = "d2m.tile_tilize_block"(%1, %2) : (memref<32x32xf32, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      d2m.pop %cb0 : <memref<32x32xf32, #l1>>
      d2m.push %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %4 = d2m.wait %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      d2m.pop %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
    }
    return
  }
}

// -----

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout1 = #ttcore.metal_layout<logical_shape = 8x8, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @generic_with_global_semaphore(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.empty() : tensor<8x8x1x1xui32, #layout1>
    %2 = d2m.create_global_semaphore(%1) {value = 0 : ui32} : tensor<8x8x1x1xui32, #layout1> -> !d2m.global_semaphore
    %3 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %stream = "d2m.stream_layout"(%arg0, %3) <{remapping = #map}> : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %4 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %stream_0 = "d2m.stream_layout"(%0, %4) <{remapping = #map}> : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %5 = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        outs(%stream_0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        additionalArgs(%2 : !d2m.global_semaphore)
     {
    // CHECK: ^unified0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %gsem0: !d2m.global_semaphore):
    ^unified0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %c1 = arith.constant 1 : index
      d2m.semaphore_wait %2, %c1 : !d2m.global_semaphore
      d2m.yield %stream_0 : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    d2m.reset_global_semaphore(%2) {value = 0 : ui32} : !d2m.global_semaphore
    return %5 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  }
}

