// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-streams -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttnn.buffer_type<l1>
#l1_1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
module {
  // CHECK-LABEL: func @test_allocate_cast
  func.func @test_allocate_cast(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    // CHECK: %[[CAST:.*]] = ttir.ttnn_metal_layout_cast {{.*}}
    // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast
    // CHECK: %[[ALLOC:.*]] = memref.alloc()
    // CHECK: %[[STREAM:.*]] = "d2m.stream_layout"(%[[CAST]], %[[ALLOC]])
    // CHECK: %[[ALLOC0:.*]] = memref.alloc()
    // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%[[CAST0]], %[[ALLOC0]])
    %cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_1>
    %0 = d2m.empty() : tensor<32x32xf32, #ttnn_layout>
    %cast_0 = ttir.ttnn_metal_layout_cast %0 : tensor<32x32xf32, #ttnn_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_1>

    // CHECK: ins(%[[STREAM]] : {{.*}})
    // CHECK: outs(%[[STREAM0]] : {{.*}})
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
      ins(%cast : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_1>)
      outs(%cast_0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_1>)  {
    ^compute0(%cb0: memref<1x1x!ttcore.tile<32x32, f32>, #l1_1>, %cb1: memref<1x1x!ttcore.tile<32x32, f32>, #l1_1>):
    }
    %cast_1 = ttir.ttnn_metal_layout_cast %cast_0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1_1> -> tensor<32x32xf32, #ttnn_layout>
    return %cast_1 : tensor<32x32xf32, #ttnn_layout>
  }
}
