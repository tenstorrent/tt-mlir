// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-attach-metal-layout --ttir-bufferization-pipeline %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>
#reduction = #tt.iterator_type<reduction>

func.func @matmul(%arg0: tensor<2x4x!tt.tile<32x32, f32>>, %arg1: tensor<4x2x!tt.tile<32x32, f32>>) -> tensor<2x2x!tt.tile<32x32, f32>> {
  // CHECK: = memref.alloc() {{.*}} : memref<2x2x!tt.tile<32x32, f32>, #l1_>
  %0 = tensor.empty() : tensor<2x2x!tt.tile<32x32, f32>>
  // CHECK: {{^  "ttir.generic".*}}
  %3 = "ttir.generic"(%arg0, %arg1, %0) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 0, 1>, operand_cb_mapping = array<i64>}> ({
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%arg2, %arg3, %arg4) : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  }) : (tensor<2x4x!tt.tile<32x32, f32>>, tensor<4x2x!tt.tile<32x32, f32>>, tensor<2x2x!tt.tile<32x32, f32>>) -> tensor<2x2x!tt.tile<32x32, f32>>
  return %3 : tensor<2x2x!tt.tile<32x32, f32>>
}

func.func @to_layout(%arg0: tensor<2x4x!tt.tile<32x32, f32>>) -> tensor<2x4x!tt.tile<32x32, f32>> {
  // CHECK: = memref.alloc() {{.*}} : memref<2x4x!tt.tile<32x32, f32>, #l1_>
  %0 = tensor.empty() : tensor<2x4x!tt.tile<32x32, f32>>
  // CHECK: {{^  "ttir.to_layout".*}}
  %3 = "ttir.to_layout"(%arg0, %0) : (tensor<2x4x!tt.tile<32x32, f32>>, tensor<2x4x!tt.tile<32x32, f32>>) -> tensor<2x4x!tt.tile<32x32, f32>>
  return %3 : tensor<2x4x!tt.tile<32x32, f32>>
}
