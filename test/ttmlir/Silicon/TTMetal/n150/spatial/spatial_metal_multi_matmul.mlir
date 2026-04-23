// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% ttnn-mode=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir

// CHECK-LABEL: func.func @multi_matmul
// CHECK: "ttmetal.enqueue_program"

#layout = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout1 = #ttcore.metal_layout<logical_shape = 128x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout2 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (0, d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
#map3 = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
module {
  func.func @multi_matmul(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>) -> (tensor<64x64xbf16>, tensor<64x64xbf16>) attributes {tt.function_type = "forward_device"} {
    %0 = d2m.empty() {virtualGridForwardMapping = #map, virtualGridInverseMapping = #map1} : tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<64x128xbf16> into tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout>
    %view = d2m.view_layout %1 remapping = #map : tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout>
    %3 = d2m.empty() {virtualGridForwardMapping = #map, virtualGridInverseMapping = #map1} : tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1>
    %4 = d2m.to_layout %arg1, %3 : tensor<128x64xbf16> into tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1>
    %view_0 = d2m.view_layout %4 remapping = #map : tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1>
    %6 = d2m.empty() {virtualGridForwardMapping = #map2, virtualGridInverseMapping = #map3} : tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout>
    %7 = d2m.to_layout %arg0, %6 : tensor<64x128xbf16> into tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout>
    %view_1 = d2m.view_layout %7 remapping = #map : tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout>
    %9 = d2m.empty() {virtualGridForwardMapping = #map2, virtualGridInverseMapping = #map3} : tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1>
    %10 = d2m.to_layout %arg1, %9 : tensor<128x64xbf16> into tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1>
    %view_2 = d2m.view_layout %10 remapping = #map : tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1>
    %12 = d2m.empty() {virtualGridForwardMapping = #map, virtualGridInverseMapping = #map1} : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>
    %view_3 = d2m.view_layout %12 remapping = #map : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2> -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>
    %14 = d2m.empty() {virtualGridForwardMapping = #map2, virtualGridInverseMapping = #map3} : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>
    %view_4 = d2m.view_layout %14 remapping = #map : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2> -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>
    %16:2 = d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%view, %view_0, %view_1, %view_2 : tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1>, tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1>)
        outs(%view_3, %view_4 : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>, tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>) {
      %21 = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map4, #map5, #map6], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
          ins(%view, %view_0 : tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1>)
          outs(%view_3 : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>) {
        %block0 = d2m.block_index(0) : index
        %block1 = d2m.block_index(1) : index
        %block2 = d2m.block_index(2) : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %22 = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, bf16>>
        %23 = d2m.remote_load %22 %view[%block0, %block2] mcast[%c0] : tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
        %24 = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, bf16>>
        %25 = d2m.remote_load %24 %view_0[%block2, %block1] mcast[%c1] : tensor<4x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<4x2x!ttcore.tile<32x32, bf16>>
        %26 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
        "d2m.tile_matmul_block"(%23, %25, %26) : (tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<4x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>) -> ()
        %27 = d2m.remote_store %view_3[%block0, %block1] %26 : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>
        d2m.yield %27 : (tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>)
      } : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>
      d2m.spatial_yield %21 : (tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>)
    }, {
      %21 = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map4, #map5, #map6], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
          ins(%view_1, %view_2 : tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1>)
          outs(%view_4 : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>) {
        %block0 = d2m.block_index(0) : index
        %block1 = d2m.block_index(1) : index
        %block2 = d2m.block_index(2) : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %22 = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, bf16>>
        %23 = d2m.remote_load %22 %view_1[%block0, %block2] mcast[%c0] : tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
        %24 = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, bf16>>
        %25 = d2m.remote_load %24 %view_2[%block2, %block1] mcast[%c1] : tensor<4x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<4x2x!ttcore.tile<32x32, bf16>>
        %26 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
        "d2m.tile_matmul_block"(%23, %25, %26) : (tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<4x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>) -> ()
        %27 = d2m.remote_store %view_4[%block0, %block1] %26 : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>
        d2m.yield %27 : (tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>)
      } : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>
      d2m.spatial_yield %21 : (tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>, tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2>
    %17 = d2m.empty() : tensor<64x64xbf16>
    %18 = d2m.to_layout %16#0, %17 : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2> into tensor<64x64xbf16> -> tensor<64x64xbf16>
    %19 = d2m.empty() : tensor<64x64xbf16>
    %20 = d2m.to_layout %16#1, %19 : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout2> into tensor<64x64xbf16> -> tensor<64x64xbf16>
    return %18, %20 : tensor<64x64xbf16>, tensor<64x64xbf16>
  }
}
