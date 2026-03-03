module {
  func.func @rand_model() -> tensor<32x32xbf16> {
    %0 = "ttir.rand"() <{dtype = bf16, high = 1.000000e+00 : f32, low = 0.000000e+00 : f32, seed = 0 : ui32, size = [32 : i32, 32 : i32]}> : () -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}
