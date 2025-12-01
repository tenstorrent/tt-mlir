module {
  func.func @slice_1d_step(%arg0: tensor<64xbf16>) -> tensor<16xbf16> {
    %1 = "ttir.slice_static"(%arg0) <{begins = [0: i32], ends = [64: i32], step = [4: i32]}> : (tensor<64xbf16>) -> tensor<16xbf16>
    return %1 : tensor<16xbf16>
  }
}
