module {
  func.func @max_pool2d_with_indices_simple(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<1x30x30x64xbf16>, %arg2: tensor<1x30x30x64xi32>) -> (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xi32>) {
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %arg1, %arg2) <{kernel = array<i32: 3, 3>, stride = array<i32: 1, 1>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x32x32x64xbf16>, tensor<1x30x30x64xbf16>, tensor<1x30x30x64xi32>) -> (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xi32>)
    return %2, %3 : tensor<1x30x30x64xbf16>, tensor<1x30x30x64xi32>
  }
}
