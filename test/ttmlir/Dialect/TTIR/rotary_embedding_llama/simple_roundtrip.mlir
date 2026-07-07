// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s
module {
  func.func @rope(%input: tensor<1x32x32x128xbf16>, %cos: tensor<1x1x32x128xbf16>,
                  %sin: tensor<1x1x32x128xbf16>, %tm: tensor<1x1x32x32xbf16>)
      -> tensor<1x32x32x128xbf16> {
    // CHECK: ttir.rotary_embedding_llama
    %0 = "ttir.rotary_embedding_llama"(%input, %cos, %sin, %tm)
        <{is_decode_mode = false}>
        : (tensor<1x32x32x128xbf16>, tensor<1x1x32x128xbf16>,
           tensor<1x1x32x128xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x32x32x128xbf16>
    return %0 : tensor<1x32x32x128xbf16>
  }
}
