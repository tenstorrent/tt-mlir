// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% ttnn-mode=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>
#ttmetal_layout = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>, undef, dram, interleaved>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>

#vgm_inv = affine_map<(d0, d1) -> (0, d0 - 2, d1)>
#vgm_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 2, d1, d2, d3)>
module  {
  // CHECK-LABEL: func.func @multi_tanh_exp
  // CHECK: "ttnn.generic"
  func.func @multi_tanh_exp(%shared_tensor: tensor<64x128xf32, #ttnn_layout>) -> (tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>) attributes {tt.function_type = "forward_device"} {
    %out_ttnn_0 = d2m.empty() : tensor<64x128xf32, #ttnn_layout>
    %out_ttnn_1 = d2m.empty() : tensor<64x128xf32, #ttnn_layout>

    %shared_tensor_metal_0 = ttir.ttnn_metal_layout_cast %shared_tensor : tensor<64x128xf32, #ttnn_layout> -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>
    %shared_tensor_metal_1 = ttir.ttnn_metal_layout_cast %shared_tensor : tensor<64x128xf32, #ttnn_layout> -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>
    %out_tensor0_metal = ttir.ttnn_metal_layout_cast %out_ttnn_0 : tensor<64x128xf32, #ttnn_layout> -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>
    %out_tensor1_metal = ttir.ttnn_metal_layout_cast %out_ttnn_1 {virtual_grid_forward_mapping = #vgm_fwd, virtual_grid_inverse_mapping = #vgm_inv} : tensor<64x128xf32, #ttnn_layout> -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>

    %result_reg_0_metal, %result_reg_1_metal = d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (1, 1)>, #ttcore.core_range<(2, 0), (3, 1)>]}
    ins(%shared_tensor_metal_0, %shared_tensor_metal_1 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>)
    outs(%out_tensor0_metal, %out_tensor1_metal : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>) {
      ^region0():
        %result_reg_0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%shared_tensor_metal_0 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>)
            outs(%out_tensor0_metal : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>)
         {
          %block0 = d2m.block_index(0) : index
          %block1 = d2m.block_index(1) : index
          %2 = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
          %reg_0_2 = d2m.remote_load %2 %shared_tensor_metal_0[%block0, %block1] : tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout> -> tensor<2x4x!ttcore.tile<32x32, f32>>
          %reg_0_3 = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
          %reg_0_4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%reg_0_2 : tensor<2x4x!ttcore.tile<32x32, f32>>) outs(%reg_0_3 : tensor<2x4x!ttcore.tile<32x32, f32>>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %reg_0_5 = "d2m.tile_tanh"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %reg_0_5 : !ttcore.tile<32x32, f32>
          } -> tensor<2x4x!ttcore.tile<32x32, f32>>
          %block0_2 = d2m.block_index(0) : index
          %block1_3 = d2m.block_index(1) : index
          %6 = d2m.remote_store %out_tensor0_metal[%block0_2, %block1_3] %reg_0_4 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>, tensor<2x4x!ttcore.tile<32x32, f32>> -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>
          d2m.yield %6 : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>)
        } : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>
        d2m.spatial_yield %result_reg_0 : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>)
    },
    {
      ^region1():
        %result_reg_1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%shared_tensor_metal_1 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>)
            outs(%out_tensor1_metal : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>)
         {
          %block0 = d2m.block_index(0) : index
          %block1 = d2m.block_index(1) : index
          %2 = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
          %reg_1_2 = d2m.remote_load %2 %shared_tensor_metal_1[%block0, %block1] : tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout> -> tensor<2x4x!ttcore.tile<32x32, f32>>
          %reg_1_3 = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
          %reg_1_4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%reg_1_2 : tensor<2x4x!ttcore.tile<32x32, f32>>) outs(%reg_1_3 : tensor<2x4x!ttcore.tile<32x32, f32>>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %reg_1_5 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %reg_1_5 : !ttcore.tile<32x32, f32>
          } -> tensor<2x4x!ttcore.tile<32x32, f32>>
          %block0_2 = d2m.block_index(0) : index
          %block1_3 = d2m.block_index(1) : index
          %6 = d2m.remote_store %out_tensor1_metal[%block0_2, %block1_3] %reg_1_4 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>, tensor<2x4x!ttcore.tile<32x32, f32>> -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>
          d2m.yield %6 : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>)
        } : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>

        d2m.spatial_yield %result_reg_1 : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>)
    }
     : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout>

    %result0 = ttir.ttnn_metal_layout_cast %result_reg_0_metal : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout> -> tensor<64x128xf32, #ttnn_layout>
    %result1 = ttir.ttnn_metal_layout_cast %result_reg_1_metal : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #ttmetal_layout> -> tensor<64x128xf32, #ttnn_layout>
    return %result0, %result1 : tensor<64x128xf32, #ttnn_layout>, tensor<64x128xf32, #ttnn_layout>
  }
}
