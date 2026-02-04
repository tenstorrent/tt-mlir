// RUN: ttmlir-opt --ttcore-register-device --ttir-bufferization-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#layout = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded, index_map = map(0)>
#layout1 = #ttcore.metal_layout<logical_shape = 128x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded, index_map = map(0)>
#layout2 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded, index_map = map(0)>
#layout3 = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded, index_map = map(0)>
#layout4 = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded, index_map = (d0, d1, d2, d3) -> (d1, d0, d2, d3)>

func.func @to_layout() -> tensor<1x2x2x2x!ttcore.tile<32x32, f32>, #layout3> {
  %arg0 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: = memref.alloc
  %0 = d2m.empty() : tensor<1x2x2x2x!ttcore.tile<32x32, f32>, #layout3>
  // CHECK: {{^  d2m.to_layout.*}}
  %1 = "d2m.to_layout"(%arg0, %0) : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x2x2x2x!ttcore.tile<32x32, f32>, #layout3>) -> tensor<1x2x2x2x!ttcore.tile<32x32, f32>, #layout3>
  return %1 : tensor<1x2x2x2x!ttcore.tile<32x32, f32>, #layout3>
}

func.func @stream_layout() -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4> {
  %arg0 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: = memref.alloc
  %0 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4>
  // CHECK: = "d2m.stream_layout"
  %stream = "d2m.stream_layout"(%arg0, %0) : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4>) -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4>
  return %stream : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4>
}

func.func @view_layout() -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4> {
  %arg0 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: = d2m.view_layout
  %view = "d2m.view_layout"(%arg0) : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4>
  return %view : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4>
}

func.func @full() -> tensor<32x32xf32> {
  // CHECK: = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
  %c = d2m.full {shape = array<i32: 32, 32>, fill_value = 1.000000e+00 : f32} : tensor<32x32xf32>
  return %c : tensor<32x32xf32>
}

#layout5 = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, interleaved, index_map=map(0)>
func.func @interleaved_tensor_memory_layout() -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout5> {
  // CHECK: memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #l1>
  %1 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout5>
  return %1 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout5>
}

// Test remote_load and remote_store bufferization - loads from a view with a different index_map
#layout_view = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded, index_map = (d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#layout_grid2x2 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded, index_map = map(0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @remote_load_bufferization
func.func @remote_load_bufferization() -> tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_grid2x2> {
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  %input = d2m.empty() : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_grid2x2>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  %output = d2m.empty() : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_grid2x2>
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %[[ALLOC0]] : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>, #l1>
  %view = d2m.view_layout %input : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_grid2x2> -> tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_view>
  // CHECK: d2m.generic
  // CHECK-NEXT: ins(%[[VIEW]] : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>, #l1>)
  // CHECK-NEXT: outs(%[[ALLOC1]] : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
  // CHECK: ^unified0(%{{.*}}: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %{{.*}}: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
  %result = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>, indexing_maps = [#map3, #map3], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%view : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_view>)
      outs(%output : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_grid2x2>)  {
  ^unified0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
    %iter0 = d2m.block_index(0) : index
    %iter1 = d2m.block_index(1) : index
    %buffer = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
    // CHECK: d2m.remote_load %{{.*}} %[[VIEW]][{{.*}}, {{.*}}] : memref<1x1x!ttcore.tile<32x32, f32>>, memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    %load_result = d2m.remote_load %buffer %view[%iter0, %iter1] : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_view> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    // CHECK: d2m.remote_store %[[ALLOC1]][{{.*}}, {{.*}}] {{.*}} : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>> -> memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %store_result = d2m.remote_store %output[%iter0, %iter1] %load_result : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_grid2x2>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_grid2x2>

    d2m.yield %store_result : (tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_grid2x2>)
  } : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_grid2x2>
  return %result : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout_grid2x2>
}
