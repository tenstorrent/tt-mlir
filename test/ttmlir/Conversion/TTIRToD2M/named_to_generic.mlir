module {
  func.func @named_elementwise(%arg0: tensor<128x96xf32>, %arg1: tensor<128x96xf32>, %arg2: tensor<128x96xf32>) -> tensor<128x96xf32> {
    %0 = "ttir.ge"(%arg0, %arg1, %arg2) : (tensor<128x96xf32>, tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
    %1 = "ttir.gelu"(%0, %arg2) : (tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
    %2 = "ttir.bitwise_not"(%1, %arg2) : (tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
    return %2 : tensor<128x96xf32>
  }
  func.func @named_reductions_R(%arg0: tensor<128x96xf32>) -> tensor<1x96xf32> {
    %0 = ttir.empty() : tensor<1x96xf32>
    %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<128x96xf32>, tensor<1x96xf32>) -> tensor<1x96xf32>
    return %1 : tensor<1x96xf32>
  }
  func.func @named_reductions_C(%arg0: tensor<128x96xf32>) -> tensor<128x1xf32> {
    %0 = ttir.empty() : tensor<128x1xf32>
    %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-1 : i32], keep_dim = true}> : (tensor<128x96xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    return %1 : tensor<128x1xf32>
  }
  func.func @named_reductions_RC(%arg0: tensor<128x96xf32>) -> tensor<1x1xf32> {
    %0 = ttir.empty() : tensor<1x1xf32>
    %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-2 : i32, -1 : i32], keep_dim = true}> : (tensor<128x96xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    return %1 : tensor<1x1xf32>
  }
  func.func @named_contractions(%arg0: tensor<128x96xf32>, %arg1: tensor<96x64xf32>, %arg2: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %0 = "ttir.matmul"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<128x96xf32>, tensor<96x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    return %0 : tensor<128x64xf32>
  }
}
