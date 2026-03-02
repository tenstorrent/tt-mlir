module {
  func.func @slice_static_0(%arg0: tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16>
    return %0 : tensor<1x8x32x64xbf16>
  }

  func.func @slice_static_1(%arg0: tensor<1x8x32x64xbf16>) -> tensor<1x8x32x32xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x32x64xbf16>) -> tensor<1x8x32x32xbf16>
    return %0 : tensor<1x8x32x32xbf16>
  }

  func.func @slice_static_2(%arg0: tensor<1x8x32x64xbf16>) -> tensor<1x8x32x32xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x32x64xbf16>) -> tensor<1x8x32x32xbf16>
    return %0 : tensor<1x8x32x32xbf16>
  }

  func.func @slice_static_3(%arg0: tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 32 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16>
    return %0 : tensor<1x8x32x64xbf16>
  }

  func.func @slice_static_4(%arg0: tensor<1x96x32x128xbf16>) -> tensor<1x96x32x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 96 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x96x32x128xbf16>) -> tensor<1x96x32x64xbf16>
    return %0 : tensor<1x96x32x64xbf16>
  }

  func.func @slice_static_5(%arg0: tensor<1x96x32x64xbf16>) -> tensor<1x96x32x32xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 96 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x96x32x64xbf16>) -> tensor<1x96x32x32xbf16>
    return %0 : tensor<1x96x32x32xbf16>
  }

  func.func @slice_static_6(%arg0: tensor<1x96x32x64xbf16>) -> tensor<1x96x32x32xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 96 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x96x32x64xbf16>) -> tensor<1x96x32x32xbf16>
    return %0 : tensor<1x96x32x32xbf16>
  }

  func.func @slice_static_7(%arg0: tensor<1x96x32x128xbf16>) -> tensor<1x96x32x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 96 : i32, 32 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x96x32x128xbf16>) -> tensor<1x96x32x64xbf16>
    return %0 : tensor<1x96x32x64xbf16>
  }
}
