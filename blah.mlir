// func.func @simple_outer_permute_31_32(%arg0: tensor<1x11x1x32xf32>) -> tensor<1x1x11x32xf32> {
//  %0 = ttir.empty() : tensor<1x1x11x32xf32>
//  %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x11x1x32xf32>, tensor<1x1x11x32xf32>) -> tensor<1x1x11x32xf32>
//  return %1 : tensor<1x1x11x32xf32>
// }

module {
  func.func @permute_with_abs(%arg0: tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32> {
    %0 = ttir.empty() : tensor<1x32x32x32xf32>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32>
    %2 = ttir.empty() : tensor<1x32x32x32xf32>
    %3 = "ttir.abs"(%1, %2) : (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32>
    return %3 : tensor<1x32x32x32xf32>
  }
}

// module {
//  func.func @permute_with_abs(%arg0: tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32> {
//    %0 = ttir.empty() : tensor<1x64x64x64xf32>
//    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x64x64x64xf32>, tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
//    return %1 : tensor<1x64x64x64xf32>
//  }
//}
