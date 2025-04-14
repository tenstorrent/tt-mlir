// RUN: ttmlir-opt --tt-register-device --ttir-bufferization-pipeline %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>
#reduction = #tt.iterator_type<reduction>
#layout = #tt.metal_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), undef, <1x1>, memref<2x4x!tt.tile<32x32, f32>, #l1_>>
#layout1 = #tt.metal_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), undef, <1x1>, memref<4x2x!tt.tile<32x32, f32>, #l1_>>
#layout2 = #tt.metal_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), undef, <1x1>, memref<2x2x!tt.tile<32x32, f32>, #l1_>>
#layout3 = #tt.metal_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), undef, <1x2>, memref<2x2x!tt.tile<32x32, f32>, #l1_>>

func.func @matmul(%arg0: tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>, %arg1: tensor<1x1x4x2x!tt.tile<32x32, f32>, #layout1>) -> tensor<1x1x2x2x!tt.tile<32x32, f32>, #layout2> {
  // CHECK: = memref.alloc(){{.*}}: memref<1x1x2x2x!tt.tile<32x32, f32>, #tt.shard<8192x4096>, #l1_>
  %0 = ttir.empty() : tensor<1x1x2x2x!tt.tile<32x32, f32>, #layout2>
  // CHECK: {{^  ttir.generic.*}}
  %1 = "ttir.generic"(%arg0, %arg1, %0) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%arg2, %arg3, %arg4) : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  }) : (tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>, tensor<1x1x4x2x!tt.tile<32x32, f32>, #layout1>, tensor<1x1x2x2x!tt.tile<32x32, f32>, #layout2>) -> tensor<1x1x2x2x!tt.tile<32x32, f32>, #layout2>
  return %1 : tensor<1x1x2x2x!tt.tile<32x32, f32>, #layout2>
}

func.func @to_layout(%arg0: tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>) -> tensor<1x2x2x2x!tt.tile<32x32, f32>, #layout3> {
  // CHECK: = memref.alloc(){{.*}}: memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.shard<8192x4096>, #l1_>
  %0 = ttir.empty() : tensor<1x2x2x2x!tt.tile<32x32, f32>, #layout3>
  // CHECK: {{^  ttir.to_layout.*}}
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>, tensor<1x2x2x2x!tt.tile<32x32, f32>, #layout3>) -> tensor<1x2x2x2x!tt.tile<32x32, f32>, #layout3>
  return %1 : tensor<1x2x2x2x!tt.tile<32x32, f32>, #layout3>
}

func.func @stream_layout(%arg0: tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>) -> tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout> {
  // CHECK: = memref.alloc(){{.*}}: memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>
  %0 = ttir.empty() : tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>
  // CHECK: = "ttir.stream_layout"
  %stream = "ttir.stream_layout"(%arg0, %0) : (tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>, tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>) -> tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>
  return %stream : tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>
}

func.func @view_layout(%arg0: tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>) -> tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout> {
  // CHECK: = "ttir.view_layout"
  %view = "ttir.view_layout"(%arg0) : (tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>) -> tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>
  return %view : tensor<1x1x2x4x!tt.tile<32x32, f32>, #layout>
}

func.func @constant() -> tensor<32x32xf32> {
  // CHECK: = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
  %c = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  return %c : tensor<32x32xf32>
}
