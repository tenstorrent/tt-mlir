module {
  func.func @ones() -> tensor<128x128xf32> {
    %0 = "ttir.ones"() <{shape = array<i32: 128, 128>}> : () -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
