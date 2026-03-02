module {
  func.func @slice_static_0(%arg0: tensor<1x32x576xbf16>) -> tensor<1x32x512xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x576xbf16>) -> tensor<1x32x512xbf16>
    return %0 : tensor<1x32x512xbf16>
  }

  func.func @slice_static_1(%arg0: tensor<8x16384x512xbf16>) -> tensor<1x16384x512xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16384 : i32, 512 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<8x16384x512xbf16>) -> tensor<1x16384x512xbf16>
    return %0 : tensor<1x16384x512xbf16>
  }

  func.func @slice_static_2(%arg0: tensor<1x32x576xbf16>) -> tensor<1x32x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 512 : i32], ends = [1 : i32, 32 : i32, 576 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x576xbf16>) -> tensor<1x32x64xbf16>
    return %0 : tensor<1x32x64xbf16>
  }

  func.func @slice_static_3(%arg0: tensor<1x32x1x32x2xf32>) -> tensor<1x32x1x32x1xf32> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 1 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1x32x2xf32>) -> tensor<1x32x1x32x1xf32>
    return %0 : tensor<1x32x1x32x1xf32>
  }

  func.func @slice_static_4(%arg0: tensor<16384x32x2xbf16>) -> tensor<32x32x2xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<16384x32x2xbf16>) -> tensor<32x32x2xbf16>
    return %0 : tensor<32x32x2xbf16>
  }

  func.func @slice_static_5(%arg0: tensor<1x32x1x32x2xbf16>) -> tensor<1x32x1x32x1xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 1 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1x32x2xbf16>) -> tensor<1x32x1x32x1xbf16>
    return %0 : tensor<1x32x1x32x1xbf16>
  }

  func.func @slice_static_6(%arg0: tensor<1x32x1x32x2xf32>) -> tensor<1x32x1x32x1xf32> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 32 : i32, 1 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1x32x2xf32>) -> tensor<1x32x1x32x1xf32>
    return %0 : tensor<1x32x1x32x1xf32>
  }

  func.func @slice_static_7(%arg0: tensor<1x32x1x32x2xbf16>) -> tensor<1x32x1x32x1xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 32 : i32, 1 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1x32x2xbf16>) -> tensor<1x32x1x32x1xbf16>
    return %0 : tensor<1x32x1x32x1xbf16>
  }

  func.func @slice_static_8(%arg0: tensor<8x16384x64xbf16>) -> tensor<1x16384x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16384 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<8x16384x64xbf16>) -> tensor<1x16384x64xbf16>
    return %0 : tensor<1x16384x64xbf16>
  }

  func.func @slice_static_9(%arg0: tensor<1x32x16x192xbf16>) -> tensor<1x32x16x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 16 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x16x192xbf16>) -> tensor<1x32x16x128xbf16>
    return %0 : tensor<1x32x16x128xbf16>
  }

  func.func @slice_static_10(%arg0: tensor<1x32x16x192xbf16>) -> tensor<1x32x16x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 128 : i32], ends = [1 : i32, 32 : i32, 16 : i32, 192 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x16x192xbf16>) -> tensor<1x32x16x64xbf16>
    return %0 : tensor<1x32x16x64xbf16>
  }

  func.func @slice_static_11(%arg0: tensor<1x32x16x32x2xf32>) -> tensor<1x32x16x32x1xf32> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 16 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x16x32x2xf32>) -> tensor<1x32x16x32x1xf32>
    return %0 : tensor<1x32x16x32x1xf32>
  }

  func.func @slice_static_12(%arg0: tensor<1x32x16x32x2xf32>) -> tensor<1x32x16x32x1xf32> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 32 : i32, 16 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x16x32x2xf32>) -> tensor<1x32x16x32x1xf32>
    return %0 : tensor<1x32x16x32x1xf32>
  }

  func.func @slice_static_13(%arg0: tensor<1x32x16x256xbf16>) -> tensor<1x32x16x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 16 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x16x256xbf16>) -> tensor<1x32x16x128xbf16>
    return %0 : tensor<1x32x16x128xbf16>
  }

  func.func @slice_static_14(%arg0: tensor<1024x3xi64>) -> tensor<1024x1xi64> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x3xi64>) -> tensor<1024x1xi64>
    return %0 : tensor<1024x1xi64>
  }

  func.func @slice_static_15(%arg0: tensor<1024x3xi64>) -> tensor<1024x1xi64> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1 : i32], ends = [1024 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x3xi64>) -> tensor<1024x1xi64>
    return %0 : tensor<1024x1xi64>
  }

  func.func @slice_static_16(%arg0: tensor<1024x3xi64>) -> tensor<1024x1xi64> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 2 : i32], ends = [1024 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x3xi64>) -> tensor<1024x1xi64>
    return %0 : tensor<1024x1xi64>
  }

  func.func @slice_static_17(%arg0: tensor<1x32x16x256xbf16>) -> tensor<1x32x16x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 128 : i32], ends = [1 : i32, 32 : i32, 16 : i32, 256 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x16x256xbf16>) -> tensor<1x32x16x128xbf16>
    return %0 : tensor<1x32x16x128xbf16>
  }

  func.func @slice_static_18(%arg0: tensor<32x2048xbf16>) -> tensor<1x2048xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [31 : i32, 0 : i32], ends = [32 : i32, 2048 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x2048xbf16>) -> tensor<1x2048xbf16>
    return %0 : tensor<1x2048xbf16>
  }
}
