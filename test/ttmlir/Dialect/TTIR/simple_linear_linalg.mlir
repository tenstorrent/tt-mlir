// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
// UNSUPPORTED: true
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "GraphModule"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<128x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<32x64xf32>) -> (tensor<32x128xf32>, tensor<32x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: = "ttir.generic"
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x64xf32>) outs(%0 : tensor<64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x128xf32>
    %2 = tensor.empty() : tensor<32x128xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %4 = linalg.matmul ins(%arg2, %1 : tensor<32x64xf32>, tensor<64x128xf32>) outs(%3 : tensor<32x128xf32>) -> tensor<32x128xf32>
    // CHECK: = "ttir.generic"
    %5 = linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %4 : tensor<128xf32>, tensor<32x128xf32>) outs(%2 : tensor<32x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.addf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<32x128xf32>
    return %5, %arg2 : tensor<32x128xf32>, tensor<32x64xf32>
  }
}
