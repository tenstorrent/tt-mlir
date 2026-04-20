// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% ttnn-mode=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#layout = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, dram, interleaved>
#layout1 = #ttcore.metal_layout<logical_shape = 128x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, dram, interleaved>
#layout2 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, l1, sharded>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0 - 1, d1 - 1)>
#vgm_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1, d2) -> (d1 - 1, d2 - 1)>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = false>
module {
  // CHECK-LABEL: func.func @multi_matmul
  // CHECK: "ttnn.generic"
  func.func @multi_matmul(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<128x64xf32, #ttnn_layout1>) -> (tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>) attributes {tt.function_type = "forward_device"} {
    %in_cast_i_0 = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x128xf32, #ttnn_layout> -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
    %in_cast_w_0 = ttir.ttnn_metal_layout_cast %arg1 : tensor<128x64xf32, #ttnn_layout1> -> tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
    %in_cast_i_1 = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x128xf32, #ttnn_layout> -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
    %in_cast_w_1 = ttir.ttnn_metal_layout_cast %arg1 : tensor<128x64xf32, #ttnn_layout1> -> tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>

    %out_0 = d2m.empty() : tensor<64x64xf32, #ttnn_layout2>
    %out_cast = ttir.ttnn_metal_layout_cast %out_0 : tensor<64x64xf32, #ttnn_layout2> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>

    %out_1 = d2m.empty() {virtualGridMapping = #map3} : tensor<64x64xf32, #ttnn_layout3>
    %out_cast_1 = ttir.ttnn_metal_layout_cast %out_1 {virtual_grid_forward_mapping = #vgm_fwd, virtual_grid_inverse_mapping = #vgm_inv}: tensor<64x64xf32, #ttnn_layout3> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>

    %result_reg_0_metal, %result_reg_1_metal = d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%in_cast_i_0, %in_cast_w_0, %in_cast_i_1, %in_cast_w_1 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>)
        outs(%out_cast, %out_cast_1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>)  {
      ^region_0:
      %result_generric_reg_0 = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
          ins(%in_cast_i_0, %in_cast_w_0 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>)
          outs(%out_cast : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>)  {
        %block0 = d2m.block_index(0) : index
        %block1 = d2m.block_index(1) : index
        %block2 = d2m.block_index(2) : index
        %4 = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
        %c0 = arith.constant 0 : index
        %5 = d2m.remote_load %4 %in_cast_i_0[%block0, %block2] mcast[%c0] : tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x4x!ttcore.tile<32x32, f32>>
        %block0_3 = d2m.block_index(0) : index
        %block1_4 = d2m.block_index(1) : index
        %block2_5 = d2m.block_index(2) : index
        %6 = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, f32>>
        %c1 = arith.constant 1 : index
        %7 = d2m.remote_load %6 %in_cast_w_0[%block2_5, %block1_4] mcast[%c1] : tensor<4x2x!ttcore.tile<32x32, f32>>, tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1> -> tensor<4x2x!ttcore.tile<32x32, f32>>
        %8 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
        %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5, %7 : tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<4x2x!ttcore.tile<32x32, f32>>) outs(%8 : tensor<2x2x!ttcore.tile<32x32, f32>>) {
        ^bb0(%in: !ttcore.tile<32x32, f32>, %in_9: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %11 = "d2m.tile_matmul"(%in, %in_9, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %11 : !ttcore.tile<32x32, f32>
        } -> tensor<2x2x!ttcore.tile<32x32, f32>>
        %block0_6 = d2m.block_index(0) : index
        %block1_7 = d2m.block_index(1) : index
        %block2_8 = d2m.block_index(2) : index
        %10 = d2m.remote_store %out_cast[%block0_6, %block1_7] %9 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>
        d2m.yield %10 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>)
      } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>
      d2m.spatial_yield %result_generric_reg_0 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>)
    }, {
      ^region_1:
      %result_generric_reg_1 = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
          ins(%in_cast_i_1, %in_cast_w_1 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>)
          outs(%out_cast_1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>)  {
        %block0 = d2m.block_index(0) : index
        %block1 = d2m.block_index(1) : index
        %block2 = d2m.block_index(2) : index
        %4 = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
        %c0 = arith.constant 0 : index
        %5 = d2m.remote_load %4 %in_cast_i_1[%block0, %block2] mcast[%c0] : tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x4x!ttcore.tile<32x32, f32>>
        %block0_3 = d2m.block_index(0) : index
        %block1_4 = d2m.block_index(1) : index
        %block2_5 = d2m.block_index(2) : index
        %6 = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, f32>>
        %c1 = arith.constant 1 : index
        %7 = d2m.remote_load %6 %in_cast_w_1[%block2_5, %block1_4] mcast[%c1] : tensor<4x2x!ttcore.tile<32x32, f32>>, tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1> -> tensor<4x2x!ttcore.tile<32x32, f32>>
        %8 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
        %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5, %7 : tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<4x2x!ttcore.tile<32x32, f32>>) outs(%8 : tensor<2x2x!ttcore.tile<32x32, f32>>) {
        ^bb0(%in: !ttcore.tile<32x32, f32>, %in_9: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %11 = "d2m.tile_matmul"(%in, %in_9, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %11 : !ttcore.tile<32x32, f32>
        } -> tensor<2x2x!ttcore.tile<32x32, f32>>
        %block0_6 = d2m.block_index(0) : index
        %block1_7 = d2m.block_index(1) : index
        %block2_8 = d2m.block_index(2) : index
        %10 = d2m.remote_store %out_cast_1[%block0_6, %block1_7] %9 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>
        d2m.yield %10 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>)
      } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>
      d2m.spatial_yield %result_generric_reg_1 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2>
    %result_reg_0_ttnn = ttir.ttnn_metal_layout_cast %result_reg_0_metal : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2> -> tensor<64x64xf32, #ttnn_layout2>
    %result_reg_1_ttnn = ttir.ttnn_metal_layout_cast %result_reg_1_metal : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout2> -> tensor<64x64xf32, #ttnn_layout3>
    return %result_reg_0_ttnn, %result_reg_1_ttnn : tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>
  }
}
