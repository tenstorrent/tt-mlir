// K projection with per-head RMS norm and cosine RoPE from Qwen3 4B
// dot_general -> reshape -> rms_norm -> permute -> multiply

module {
  func.func @k_proj_rms_rope(%arg0: tensor<576x2560xbf16>, %arg1: tensor<2560x1024xbf16>, %arg2: tensor<128xbf16>, %arg3: tensor<32x8x18x128xbf16>) -> tensor<32x8x18x128xbf16> {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x2560xbf16>, tensor<2560x1024xbf16>) -> tensor<576x1024xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [32 : i32, 18 : i32, 8 : i32, 128 : i32]}> : (tensor<576x1024xbf16>) -> tensor<32x18x8x128xbf16>
    %2 = "ttir.rms_norm"(%1, %arg2) <{epsilon = 9.99999997E-7 : f32, normalized_shape = array<i64: 128>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x18x8x128xbf16>, tensor<128xbf16>) -> tensor<32x18x8x128xbf16>
    %3 = "ttir.permute"(%2) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x18x8x128xbf16>) -> tensor<32x8x18x128xbf16>
    %4 = "ttir.multiply"(%3, %arg3) : (tensor<32x8x18x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x8x18x128xbf16>
    return %4 : tensor<32x8x18x128xbf16>
  }
}
