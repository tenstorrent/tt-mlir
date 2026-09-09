// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %s | FileCheck %s

// RoPEDecodeFusing must recognize the decode layout transform (BHSD -> SBHD,
// seq == 1) whether it is expressed as permute [2,0,1,3] or as the equivalent
// layout-preserving reshape that the pre-fusing canonicalizer folds it into.
//
// Here the RoPE output is reshaped (not permuted) to [1,32,8,64]. The decode
// fusing must still fire, producing a decode-mode rotary_embedding (token_index).

module {
  // CHECK-LABEL: @rope_decode_with_output_reshape
  // CHECK: "ttnn.rotary_embedding"
  // CHECK-SAME: token_index
  func.func @rope_decode_with_output_reshape(%x: tensor<32x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x32x8x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %rope = "ttir.add"(%x_cos, %rot_sin) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>

    // Layout-preserving reshape BHSD -> SBHD (seq == 1): the canonicalized form
    // of permute [2,0,1,3].
    %result = "ttir.reshape"(%rope) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<1x32x8x64xbf16>
    return %result : tensor<1x32x8x64xbf16>
  }
}
