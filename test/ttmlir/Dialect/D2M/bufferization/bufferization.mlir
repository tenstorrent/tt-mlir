// RUN: ttmlir-opt --ttcore-register-device --ttir-bufferization-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#layout = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1>
#layout1 = #ttcore.metal_layout<logical_shape = 128x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1>
#layout2 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1>
#layout3 = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1>
#layout4 = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, index_map = (d0, d1, d2, d3) -> (d1, d0, d2, d3)>

func.func @matmul() -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2> {
  %arg0 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
  %arg1 = d2m.empty() : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
  // CHECK: = memref.alloc
  // CHECK: = memref.alloc
  // CHECK: = memref.alloc
  %0 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: {{^  d2m.generic.*}}
  %1 = "d2m.generic"(%arg0, %arg1, %0) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    "d2m.tile_matmul_block"(%arg2, %arg3, %arg4) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    d2m.yield %arg4 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  }) : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>
  return %1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>
}

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

func.func @constant() -> tensor<32x32xf32> {
  // CHECK: = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
  %c = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  return %c : tensor<32x32xf32>
}
