// RUN: ttmlir-opt %s --ttir-elementwise-fusion --linalg-fuse-elementwise-ops -canonicalize -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// Build a ttir.generic whose compute region contains two linalg.generics:
// 1) multiply, 2) add. The linalg fusion pass should fuse them into one.
func.func @nested_linalg_fuse(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %out = ttir.empty() : tensor<4x4xf32>
  %res = ttir.generic {grid = #ttcore.grid<1x1>, block_factors = [1, 1], indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>]}
      ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%out : tensor<4x4xf32>)  {
  ^compute(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %o: tensor<4x4xf32>):
    %tmp0 = tensor.empty() : tensor<4x4xf32>
    %mul = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%tmp0 : tensor<4x4xf32>) {
        ^bb0(%x: f32, %y: f32, %z: f32):
          %m = arith.mulf %x, %y : f32
          linalg.yield %m : f32
      } -> tensor<4x4xf32>
    %tmp1 = tensor.empty() : tensor<4x4xf32>
    %add = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%mul, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%tmp1 : tensor<4x4xf32>) {
        ^bb0(%u: f32, %v: f32, %w: f32):
          %s = arith.addf %u, %v : f32
          linalg.yield %s : f32
      } -> tensor<4x4xf32>
    // Yield the region argument to respect TTIR contract
    ttir.yield %add : (tensor<4x4xf32>)
  } : tensor<4x4xf32>
  return %res : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @nested_linalg_fuse
// CHECK: ttir.generic
// CHECK: linalg.generic

func.func @nested_generic_fuse(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %out0 = ttir.empty() : tensor<4x4xf32>
  %res0 = ttir.generic {grid = #ttcore.grid<1x1>, block_factors = [1, 1], indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>]}
      ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%out0 : tensor<4x4xf32>)  {
  ^compute(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %o: tensor<4x4xf32>):
    %tmp0 = tensor.empty() : tensor<4x4xf32>
    %mul = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%tmp0 : tensor<4x4xf32>) {
        ^bb0(%x: f32, %y: f32, %z: f32):
          %m = arith.mulf %x, %y : f32
          linalg.yield %m : f32
      } -> tensor<4x4xf32>
    ttir.yield %mul : (tensor<4x4xf32>)
  } : tensor<4x4xf32>
  %out1 = ttir.empty() : tensor<4x4xf32>
  %res1 = ttir.generic {grid = #ttcore.grid<1x1>, block_factors = [1, 1], indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>]}
      ins(%res0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%out1 : tensor<4x4xf32>)  {
  ^compute(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %o: tensor<4x4xf32>):
    %tmp0 = tensor.empty() : tensor<4x4xf32>
    %add = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%tmp0 : tensor<4x4xf32>) {
        ^bb0(%u: f32, %v: f32, %w: f32):
          %s = arith.addf %u, %v : f32
          linalg.yield %s : f32
      } -> tensor<4x4xf32>
    // Yield the region argument to respect TTIR contract
    ttir.yield %add : (tensor<4x4xf32>)
  } : tensor<4x4xf32>
  return %res1 : tensor<4x4xf32>
}
