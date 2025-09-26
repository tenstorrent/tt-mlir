// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = d2m.empty() : tensor<64x128xf32>
  %1 = "d2m.generic"(%arg0, %arg1, %0) <{
    block_factors = [1, 1],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [#map, #map, #map],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<compute>],
    operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg2: memref<64x128xf32>, %arg3: memref<64x128xf32>, %arg4: memref<64x128xf32>):
    // lit CHECK to make sure this constant stays inside the generic region
    // CHECK: d2m.generic
    // CHECK: arith.constant 0 : index
    %i = arith.constant 0 : index
    %2 = arith.constant 0.000000e+00 : f32
    memref.store %2, %arg2[%i, %i] : memref<64x128xf32>
    "d2m.yield"(%arg3) : (memref<64x128xf32>) -> ()
  }) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
