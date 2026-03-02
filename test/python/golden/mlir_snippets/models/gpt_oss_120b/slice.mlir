module {
  func.func @slice_static_0(%arg0: tensor<1x16x128x64xbf16>) -> tensor<1x16x128x32xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 128 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x128x64xbf16>) -> tensor<1x16x128x32xbf16>
    return %0 : tensor<1x16x128x32xbf16>
  }

  func.func @slice_static_1(%arg0: tensor<1x16x128x64xbf16>) -> tensor<1x16x128x32xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 16 : i32, 128 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x128x64xbf16>) -> tensor<1x16x128x32xbf16>
    return %0 : tensor<1x16x128x32xbf16>
  }

  func.func @slice_static_2(%arg0: tensor<1x2x1x128x64xbf16>) -> tensor<1x2x1x128x32xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 2 : i32, 1 : i32, 128 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x1x128x64xbf16>) -> tensor<1x2x1x128x32xbf16>
    return %0 : tensor<1x2x1x128x32xbf16>
  }

  func.func @slice_static_3(%arg0: tensor<1x2x1x128x64xbf16>) -> tensor<1x2x1x128x32xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 2 : i32, 1 : i32, 128 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x1x128x64xbf16>) -> tensor<1x2x1x128x32xbf16>
    return %0 : tensor<1x2x1x128x32xbf16>
  }

  func.func @slice_static_4(%arg0: tensor<4x16xbf16>) -> tensor<1x16xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [1 : i32, 16 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x16xbf16>) -> tensor<1x16xbf16>
    return %0 : tensor<1x16xbf16>
  }

  func.func @slice_static_5(%arg0: tensor<1x16x128x129xbf16>) -> tensor<1x16x128x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 128 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x128x129xbf16>) -> tensor<1x16x128x128xbf16>
    return %0 : tensor<1x16x128x128xbf16>
  }

  func.func @slice_static_6(%arg0: tensor<32x128x5760xbf16>) -> tensor<32x128x2880xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 1 : i32], ends = [32 : i32, 128 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 2 : i32]}> : (tensor<32x128x5760xbf16>) -> tensor<32x128x2880xbf16>
    return %0 : tensor<32x128x2880xbf16>
  }

  func.func @slice_static_7(%arg0: tensor<32x128x5760xbf16>) -> tensor<32x128x2880xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 128 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 2 : i32]}> : (tensor<32x128x5760xbf16>) -> tensor<32x128x2880xbf16>
    return %0 : tensor<32x128x2880xbf16>
  }

  func.func @slice_static_8(%arg0: tensor<128x128xi32>) -> tensor<128x4xi32> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [128 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128x128xi32>) -> tensor<128x4xi32>
    return %0 : tensor<128x4xi32>
  }

  func.func @slice_static_9(%arg0: tensor<128x128xbf16>) -> tensor<128x4xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [128 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128x128xbf16>) -> tensor<128x4xbf16>
    return %0 : tensor<128x4xbf16>
  }

  func.func @slice_static_10(%arg0: tensor<128x4x2xi64>) -> tensor<128x4x1xi64> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [128 : i32, 4 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<128x4x2xi64>) -> tensor<128x4x1xi64>
    return %0 : tensor<128x4x1xi64>
  }

  func.func @slice_static_11(%arg0: tensor<128x4x2xi64>) -> tensor<128x4x1xi64> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 1 : i32], ends = [128 : i32, 4 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<128x4x2xi64>) -> tensor<128x4x1xi64>
    return %0 : tensor<128x4x1xi64>
  }

  func.func @slice_static_12(%arg0: tensor<128x4x32xbf16>) -> tensor<128x1x32xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [128 : i32, 1 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<128x4x32xbf16>) -> tensor<128x1x32xbf16>
    return %0 : tensor<128x1x32xbf16>
  }
}
