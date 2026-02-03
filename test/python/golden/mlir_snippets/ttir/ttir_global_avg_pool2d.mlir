module attributes {} {
  func.func @global_avg_pool2d(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x1x1x32xbf16> {
    %0 = "ttir.global_avg_pool2d"(%arg0) : (tensor<1x128x128x32xbf16>) -> tensor<1x1x1x32xbf16>
    return %0 : tensor<1x1x1x32xbf16>
  }
}
