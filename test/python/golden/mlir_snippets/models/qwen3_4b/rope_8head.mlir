// RoPE (Rotary Position Embedding) for 8-head K in Qwen3 4B attention
// slice -> neg -> concat -> multiply -> add

module {
  func.func @rope_8head(%arg0: tensor<32x8x18x128xbf16>, %arg1: tensor<32x8x18x128xbf16>, %arg2: tensor<32x8x18x128xbf16>) -> tensor<32x8x18x128xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [32 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<32x8x18x64xbf16>
    %1 = "ttir.neg"(%0) : (tensor<32x8x18x64xbf16>) -> tensor<32x8x18x64xbf16>
    %2 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 18 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<32x8x18x64xbf16>
    %3 = "ttir.concat"(%1, %2) <{dim = 3 : si32}> : (tensor<32x8x18x64xbf16>, tensor<32x8x18x64xbf16>) -> tensor<32x8x18x128xbf16>
    %4 = "ttir.multiply"(%3, %arg2) : (tensor<32x8x18x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x8x18x128xbf16>
    %5 = "ttir.multiply"(%arg0, %arg1) : (tensor<32x8x18x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x8x18x128xbf16>
    %6 = "ttir.add"(%5, %4) : (tensor<32x8x18x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x8x18x128xbf16>
    return %6 : tensor<32x8x18x128xbf16>
  }
}
