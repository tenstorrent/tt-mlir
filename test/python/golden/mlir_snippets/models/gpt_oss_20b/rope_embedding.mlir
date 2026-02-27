// TTIR subgraph from GPT OSS: slice (head dim) -> rotary (cos/sin) -> concat
// Inputs: qkv_16x8x1x64, cos (16x8x1x32), sin (16x8x1x32)

module {
  func.func @gpt_oss_20b(
      %arg0: tensor<16x8x1x64xbf16>,
      %arg1: tensor<16x8x1x32xbf16>,
      %arg2: tensor<16x8x1x32xbf16>)
      -> tensor<16x8x1x64xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [16 : i32, 8 : i32, 1 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<16x8x1x64xbf16>) -> tensor<16x8x1x32xbf16>
    %1 = "ttir.multiply"(%0, %arg1) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %2 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [16 : i32, 8 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<16x8x1x64xbf16>) -> tensor<16x8x1x32xbf16>
    %3 = "ttir.multiply"(%2, %arg2) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %4 = "ttir.subtract"(%1, %3) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %5 = "ttir.multiply"(%2, %arg1) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %6 = "ttir.multiply"(%0, %arg2) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %7 = "ttir.add"(%5, %6) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %8 = "ttir.concat"(%4, %7) <{dim = 3 : si32}> : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x64xbf16>
    return %8 : tensor<16x8x1x64xbf16>
  }
}
