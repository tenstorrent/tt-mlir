module {
  func.func @sum_0(%arg0: tensor<32x17x5120xf32>) -> tensor<32x17xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x17x5120xf32>) -> tensor<32x17xf32>
    return %0 : tensor<32x17xf32>
  }

  func.func @sum_1(%arg0: tensor<32x17x1x128xf32>) -> tensor<32x17x1xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x17x1x128xf32>) -> tensor<32x17x1xf32>
    return %0 : tensor<32x17x1xf32>
  }

  func.func @sum_2(%arg0: tensor<32x17x8x128xf32>) -> tensor<32x17x8xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x17x8x128xf32>) -> tensor<32x17x8xf32>
    return %0 : tensor<32x17x8xf32>
  }

  func.func @sum_3(%arg0: tensor<32x8x17x128xf32>) -> tensor<32x8x17xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x8x17x128xf32>) -> tensor<32x8x17xf32>
    return %0 : tensor<32x8x17xf32>
  }
}
