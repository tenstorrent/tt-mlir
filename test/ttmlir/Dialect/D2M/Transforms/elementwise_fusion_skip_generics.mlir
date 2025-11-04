// RUN: ttmlir-opt --d2m-elementwise-fusion %s | FileCheck %s

#layout = #ttcore.metal_layout<logical_shape = 1x32x32x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
module {
  func.func @skip_fusion_on_empty_compute(%arg0: tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32> {
    %0 = d2m.empty() : tensor<1x1x1x1x1x32x32x32xf32, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<1x32x32x32xf32> into tensor<1x1x1x1x1x32x32x32xf32, #layout> -> tensor<1x1x1x1x1x32x32x32xf32, #layout>
    // CHECK-LABEL: func.func @skip_fusion_on_empty_compute
    // CHECK: d2m.generic {{.*}} threads = [#d2m.thread<compute>]
    // CHECK: d2m.wait
    // CHECK: d2m.yield
    %2 = d2m.generic {block_factors = [1, 1, 1, 1], grid = #ttcore.grid<1x1x1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel, #parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%1 : tensor<1x1x1x1x1x32x32x32xf32, #layout>)
        outs(%1 : tensor<1x1x1x1x1x32x32x32xf32, #layout>)  {
    ^compute0(%cb0: !d2m.cb<tensor<1x32x32x32xf32>>, %cb1: !d2m.cb<tensor<1x32x32x32xf32>>):
      %5 = d2m.wait %cb0 : <tensor<1x32x32x32xf32>> -> tensor<1x32x32x32xf32>
      d2m.yield %5 : (tensor<1x32x32x32xf32>)
    } : tensor<1x1x1x1x1x32x32x32xf32, #layout>
    %3 = d2m.empty() : tensor<1x32x32x32xf32>
    %4 = d2m.to_layout %2, %3 : tensor<1x1x1x1x1x32x32x32xf32, #layout> into tensor<1x32x32x32xf32> -> tensor<1x32x32x32xf32>
    return %4 : tensor<1x32x32x32xf32>
  }
}
