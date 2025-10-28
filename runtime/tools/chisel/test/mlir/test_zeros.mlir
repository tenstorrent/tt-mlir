module {
  func.func @zeros() -> tensor<128x128xbf16> {
    %0 = ttir.empty() : tensor<128x128xbf16>
    %1 = "ttir.zeros"() <{shape = array<i32: 128, 128>}> : () -> tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }
}
