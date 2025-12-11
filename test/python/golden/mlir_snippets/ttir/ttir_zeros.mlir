module {
  func.func @zeros() -> tensor<128x128xf32> {
    %0 = "ttir.zeros"() <{shape = array<i32: 128, 128>}> : () -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
