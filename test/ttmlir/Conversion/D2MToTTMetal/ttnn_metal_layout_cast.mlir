// RUN: ttmlir-opt --ttir-bufferization-pipeline="ttnn-mode=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>
#ttnn_l1_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1, (d0, d1) -> (0, d0, d1)>,
  memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>
  >
#metal_layout = #ttcore.metal_layout<
  logical_shape = 32x32,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1
  >

module {
  // CHECK-NOT: %arg0: memref
  func.func @test_bufferization(%arg0: tensor<32x32xf32, #ttnn_l1_layout>) -> tensor<32x32xf32, #ttnn_l1_layout> {
    // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_layout> -> memref<
    %2 = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_l1_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #metal_layout>

    // CHECK-NOT: memref.alloc()
    %3 =  d2m.empty() : tensor<32x32xf32, #ttnn_l1_layout>

    // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %{{[0-9]+}} : tensor<32x32xf32, #ttnn_layout> -> memref
    %4 = ttir.ttnn_metal_layout_cast %3 : tensor<32x32xf32, #ttnn_l1_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #metal_layout>

    // CHECK: ins(%[[CAST0]] : memref<{{.*}}>)
    // CHECK: outs(%[[CAST1]] : memref<{{.*}}>)
    d2m.generic {grid = #ttcore.grid<1x1>,
                  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                  iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
                  block_factors = [1, 1],
                  threads = [#d2m.thread<compute>]}
        ins(%2 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #metal_layout>)
        outs(%4 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #metal_layout>) {
      ^compute0(%arg_in: tensor<1x1x!ttcore.tile<32x32, f32>>, %arg_out: tensor<1x1x!ttcore.tile<32x32, f32>>):
        d2m.yield %arg_in : (tensor<1x1x!ttcore.tile<32x32, f32>>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #metal_layout>

    // CHECK: ttir.ttnn_metal_layout_cast %[[CAST1]] : memref{{.*}}> -> tensor<32x32xf32, #ttnn_layout>
    %5 = ttir.ttnn_metal_layout_cast %4 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #metal_layout> -> tensor<32x32xf32, #ttnn_l1_layout>

    return %5 : tensor<32x32xf32, #ttnn_l1_layout>
  }
}
