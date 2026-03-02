module {
  func.func @slice_static_0(%arg0: tensor<32x1x17x128xbf16>) -> tensor<32x1x17x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [32 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<32x1x17x64xbf16>
    return %0 : tensor<32x1x17x64xbf16>
  }

  func.func @slice_static_1(%arg0: tensor<32x1x17x128xbf16>) -> tensor<32x1x17x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 1 : i32, 17 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<32x1x17x64xbf16>
    return %0 : tensor<32x1x17x64xbf16>
  }

  func.func @slice_static_2(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_3(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [1 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_4(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [2 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [3 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_5(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [3 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [4 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_6(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [4 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [5 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_7(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [5 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_8(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [6 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [7 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_9(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [7 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_10(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [8 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [9 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_11(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [9 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [10 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_12(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [10 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [11 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_13(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [11 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [12 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_14(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [12 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [13 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_15(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [13 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [14 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_16(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [14 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [15 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_17(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [15 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [16 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_18(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [16 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [17 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_19(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [17 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [18 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_20(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [18 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [19 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_21(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [19 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [20 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_22(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [20 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [21 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_23(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [21 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [22 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_24(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [22 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [23 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_25(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [23 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [24 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_26(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [24 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [25 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_27(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [25 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [26 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_28(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [26 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [27 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_29(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [27 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [28 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_30(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [28 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [29 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_31(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [29 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [30 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_32(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [30 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [31 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_33(%arg0: tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [31 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 1 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %0 : tensor<1x1x17x128xbf16>
  }

  func.func @slice_static_34(%arg0: tensor<32x8x17x128xbf16>) -> tensor<32x8x17x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [32 : i32, 8 : i32, 17 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x17x128xbf16>) -> tensor<32x8x17x64xbf16>
    return %0 : tensor<32x8x17x64xbf16>
  }

  func.func @slice_static_35(%arg0: tensor<32x8x17x128xbf16>) -> tensor<32x8x17x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 17 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x17x128xbf16>) -> tensor<32x8x17x64xbf16>
    return %0 : tensor<32x8x17x64xbf16>
  }
}
