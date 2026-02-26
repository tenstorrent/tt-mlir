module {
  func.func @slice_reshape_16x576_to_16x1x1x512_bf16(%arg0: tensor<16x576xbf16>) -> tensor<16x1x1x512xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [16 : i32, 512 : i32], step = [1 : i32, 1 : i32]}> : (tensor<16x576xbf16>) -> tensor<16x512xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [16 : i32, 1 : i32, 1 : i32, 512 : i32]}> : (tensor<16x512xbf16>) -> tensor<16x1x1x512xbf16>
    return %1 : tensor<16x1x1x512xbf16>
  }
}
