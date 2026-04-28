module {
  func.func @injected(%arg0: tensor<32x32xf32>) -> tensor<64x64xf32> {
    // NOTE: copy the tensor to the right size to return.
    %horizontal = "ttir.concat"(%arg0, %arg0) <{dim = 0 : si32}>
        : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %vertical = "ttir.concat"(%horizontal, %horizontal) <{dim = 1 : si32}>
        : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
    return %vertical : tensor<64x64xf32>
  }
}
