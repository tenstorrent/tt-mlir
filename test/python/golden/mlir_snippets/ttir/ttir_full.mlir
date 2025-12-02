module {
  func.func @full_float() -> tensor<64x128xbf16> {
    %0 = "ttir.full"() <{shape = array<i32: 64, 128>, fill_value = 3.0 : f32}> : () -> tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}
