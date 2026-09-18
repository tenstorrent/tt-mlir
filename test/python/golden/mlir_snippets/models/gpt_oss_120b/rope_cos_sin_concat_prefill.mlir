// RoPE cos+sin+concat subgraph (prefill shape: 1x17x32 -> 1x17x64).
// Builds freqs via typecast(i32->f32) -> reshape -> matmul -> permute, then
// parallel cos/sin branches each scaled in f32, typecast to bf16, and
// concatenated on the last dim. Trace hoist fails here because the D2M
// lowering of the cross-type typecast + concat fanout materializes a
// non-hoistable ttnn.empty interleaved between hoistable ops (issue #8402).

module {
  func.func @rope_cos_sin_concat_prefill(
      %position_ids : tensor<17xi32>,
      %inv_freq     : tensor<1x32x1xf32>,
      %scale_cos    : tensor<1x1x1xf32>,
      %scale_sin    : tensor<1x1x1xf32>)
      -> tensor<1x17x64xbf16> {
    %0 = "ttir.typecast"(%position_ids) : (tensor<17xi32>) -> tensor<17xf32>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 17 : i32]}> : (tensor<17xf32>) -> tensor<1x1x17xf32>
    %2 = "ttir.matmul"(%inv_freq, %1) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x1xf32>, tensor<1x1x17xf32>) -> tensor<1x32x17xf32>
    %3 = "ttir.permute"(%2) <{permutation = array<i64: 0, 2, 1>}> : (tensor<1x32x17xf32>) -> tensor<1x17x32xf32>
    %4 = "ttir.cos"(%3) : (tensor<1x17x32xf32>) -> tensor<1x17x32xf32>
    %5 = "ttir.multiply"(%4, %scale_cos) : (tensor<1x17x32xf32>, tensor<1x1x1xf32>) -> tensor<1x17x32xf32>
    %6 = "ttir.sin"(%3) : (tensor<1x17x32xf32>) -> tensor<1x17x32xf32>
    %7 = "ttir.multiply"(%6, %scale_sin) : (tensor<1x17x32xf32>, tensor<1x1x1xf32>) -> tensor<1x17x32xf32>
    %8 = "ttir.typecast"(%5) : (tensor<1x17x32xf32>) -> tensor<1x17x32xbf16>
    %9 = "ttir.typecast"(%7) : (tensor<1x17x32xf32>) -> tensor<1x17x32xbf16>
    %10 = "ttir.concat"(%8, %9) <{dim = 2 : si32}> : (tensor<1x17x32xbf16>, tensor<1x17x32xbf16>) -> tensor<1x17x64xbf16>
    return %10 : tensor<1x17x64xbf16>
  }
}
