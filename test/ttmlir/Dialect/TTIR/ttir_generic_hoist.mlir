// RUN: ttmlir-opt --canonicalize %s | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>
func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = "ttir.generic"(%arg0, %arg1, %0) <{
    grid = #tt.grid<1x1>,
    indexing_maps = [#map, #map, #map],
    iterator_types = [#parallel, #parallel],
    operandSegmentSizes = array<i32: 2, 0, 1>,
    operand_cb_mapping = array<i64>}> ({
  ^bb0(%arg2: memref<64x128xf32>, %arg3: tensor<64x128xf32>, %arg4: tensor<64x128xf32>):
    // lit CHECK to make sure this constant stays inside the generic region
    // CHECK: ttir.generic
    // CHECK: arith.constant 0 : index
    %i = arith.constant 0 : index
    %2 = arith.constant 0.000000e+00 : f32
    memref.store %2, %arg2[%i, %i] : memref<64x128xf32>
    "ttir.yield"(%arg3) : (tensor<64x128xf32>) -> ()
  }) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
