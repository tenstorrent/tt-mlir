// RUN: ttmlir-opt %s -ttir-elementwise-fusion -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %out0 = ttir.empty() : tensor<4x4xf32>
    %prod = ttir.generic {grid = #ttcore.grid<1x1>, block_factors = [1, 1], indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>]}
        ins(%arg0 : tensor<4x4xf32>)
        outs(%out0 : tensor<4x4xf32>)  {
    ^compute(%t0: tensor<4x4xf32>, %t1: tensor<4x4xf32>):
      ttir.yield %t0 : (tensor<4x4xf32>)
    } : tensor<4x4xf32>
    %out1 = ttir.empty() : tensor<4x4xf32>
    %cons = ttir.generic {grid = #ttcore.grid<1x1>, block_factors = [1, 1], indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>]}
        ins(%prod, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
        outs(%out1 : tensor<4x4xf32>)  {
    ^compute(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>):
      ttir.yield %a : (tensor<4x4xf32>)
    } : tensor<4x4xf32>
    return %cons : tensor<4x4xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: ttir.generic {{.*}} threads = [#ttir.thread<compute>]
// CHECK-NOT: ttir.generic {{.*}} threads = [#ttir.thread<compute>]
