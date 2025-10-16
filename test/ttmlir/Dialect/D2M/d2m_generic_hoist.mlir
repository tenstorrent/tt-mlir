// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = d2m.empty() : tensor<64x128xf32>
  %1 = d2m.generic {
    block_factors = [1, 1],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [#map, #map, #map],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<compute>]}
    ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>)
    outs(%0 : tensor<64x128xf32>) {
  ^bb0(%cb2: !d2m.cb<tensor<64x128xf32>>, %cb3: !d2m.cb<tensor<64x128xf32>>, %cb4: !d2m.cb<tensor<64x128xf32>>):
    %arg2 = d2m.pop %cb2 : !d2m.cb<tensor<64x128xf32>> -> tensor<64x128xf32>
    %arg3 = d2m.pop %cb3 : !d2m.cb<tensor<64x128xf32>> -> tensor<64x128xf32>
    %arg4 = d2m.reserve %cb4 : !d2m.cb<tensor<64x128xf32>> -> tensor<64x128xf32>
    // lit CHECK to make sure this constant stays inside the generic region
    // CHECK: d2m.generic
    // CHECK: arith.constant 0 : index
    %i = arith.constant 0 : index
    %2 = arith.constant 0.000000e+00 : f32
    // Use the constant in a meaningful way
    %extracted = tensor.extract %arg2[%i, %i] : tensor<64x128xf32>
    %added = arith.addf %extracted, %2 : f32
    %result = tensor.insert %added into %arg4[%i, %i] : tensor<64x128xf32>
    d2m.yield %result : (tensor<64x128xf32>)
  } : tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
