// RoPE (Rotary Position Embedding) fusing tests extracted from real model IRs.
//
// Source: test/ttmlir/models/single_blocks_and_layers/*_layer.mlir
//
// Tests cover model-specific shape configurations:
//   1. Llama 3.2 1B decode: batch=32, 8 heads, head_dim=64
//   2. Llama 3.2 1B prefill: batch=32, 8 heads, seq=18, head_dim=64
//   3. Llama 3.2 3B decode: batch=32, 8 heads, head_dim=128
//   4. Falcon 3 1B decode: batch=32, 4 heads, head_dim=256
//   5. Decode with output permute [2,0,1,3] (KV-head cache pattern)

// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true" %s | FileCheck %s

module {

  // Llama 3.2 1B decode: batch=32, 8 KV-heads, seq=1, head_dim=64.
  // CHECK-LABEL: @rope_llama_1b_decode
  // CHECK: "ttnn.rotary_embedding"
  func.func @rope_llama_1b_decode(%x: tensor<32x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    return %result : tensor<32x8x1x64xbf16>
  }

  // Llama 3.2 1B prefill: batch=32, 8 KV-heads, seq=18, head_dim=64.
  // CHECK-LABEL: @rope_llama_1b_prefill
  // CHECK: "ttnn.rotary_embedding"
  func.func @rope_llama_1b_prefill(%x: tensor<32x8x18x64xbf16>, %cos: tensor<1x1x18x64xbf16>, %sin: tensor<1x1x18x64xbf16>) -> tensor<32x8x18x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x18x64xbf16>) -> tensor<32x8x18x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<32x8x18x64xbf16>, tensor<32x8x18x64xbf16>) -> tensor<32x8x18x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [32 : i32, 8 : i32, 18 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x64xbf16>) -> tensor<32x8x18x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<32x8x18x32xbf16>) -> tensor<32x8x18x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 18 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x64xbf16>) -> tensor<32x8x18x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<32x8x18x32xbf16>, tensor<32x8x18x32xbf16>) -> tensor<32x8x18x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x18x64xbf16>) -> tensor<32x8x18x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<32x8x18x64xbf16>, tensor<32x8x18x64xbf16>) -> tensor<32x8x18x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<32x8x18x64xbf16>, tensor<32x8x18x64xbf16>) -> tensor<32x8x18x64xbf16>
    return %result : tensor<32x8x18x64xbf16>
  }

  // Llama 3.2 3B decode: batch=32, 8 KV-heads, seq=1, head_dim=128.
  // CHECK-LABEL: @rope_llama_3b_decode
  // CHECK: "ttnn.rotary_embedding"
  func.func @rope_llama_3b_decode(%x: tensor<32x8x1x128xbf16>, %cos: tensor<1x1x1x128xbf16>, %sin: tensor<1x1x1x128xbf16>) -> tensor<32x8x1x128xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x128xbf16>) -> tensor<32x8x1x128xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<32x8x1x128xbf16>, tensor<32x8x1x128xbf16>) -> tensor<32x8x1x128xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x128xbf16>) -> tensor<32x8x1x64xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x128xbf16>) -> tensor<32x8x1x64xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x128xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x128xbf16>) -> tensor<32x8x1x128xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<32x8x1x128xbf16>, tensor<32x8x1x128xbf16>) -> tensor<32x8x1x128xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<32x8x1x128xbf16>, tensor<32x8x1x128xbf16>) -> tensor<32x8x1x128xbf16>
    return %result : tensor<32x8x1x128xbf16>
  }

  // Falcon 3 1B decode: batch=32, 4 KV-heads, seq=1, head_dim=256.
  // CHECK-LABEL: @rope_falcon_decode
  // CHECK: "ttnn.rotary_embedding"
  func.func @rope_falcon_decode(%x: tensor<32x4x1x256xbf16>, %cos: tensor<1x1x1x256xbf16>, %sin: tensor<1x1x1x256xbf16>) -> tensor<32x4x1x256xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 32, 4, 1, 1>}> : (tensor<1x1x1x256xbf16>) -> tensor<32x4x1x256xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<32x4x1x256xbf16>, tensor<32x4x1x256xbf16>) -> tensor<32x4x1x256xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 128 : i32], ends = [32 : i32, 4 : i32, 1 : i32, 256 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x4x1x256xbf16>) -> tensor<32x4x1x128xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<32x4x1x128xbf16>) -> tensor<32x4x1x128xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 4 : i32, 1 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x4x1x256xbf16>) -> tensor<32x4x1x128xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<32x4x1x128xbf16>, tensor<32x4x1x128xbf16>) -> tensor<32x4x1x256xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 32, 4, 1, 1>}> : (tensor<1x1x1x256xbf16>) -> tensor<32x4x1x256xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<32x4x1x256xbf16>, tensor<32x4x1x256xbf16>) -> tensor<32x4x1x256xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<32x4x1x256xbf16>, tensor<32x4x1x256xbf16>) -> tensor<32x4x1x256xbf16>
    return %result : tensor<32x4x1x256xbf16>
  }

  // Decode with output permute: batch=32, 8 KV-heads, head_dim=64.
  // Models permute RoPE output [2,0,1,3] for KV cache storage (BHSD -> SBHD).
  // CHECK-LABEL: @rope_decode_with_output_permute
  // CHECK: "ttnn.rotary_embedding"
  func.func @rope_decode_with_output_permute(%x: tensor<32x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x32x8x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %rope = "ttir.add"(%x_cos, %rot_sin) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>

    // Permute BHSD -> SBHD for KV cache storage.
    %result = "ttir.permute"(%rope) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<32x8x1x64xbf16>) -> tensor<1x32x8x64xbf16>
    return %result : tensor<1x32x8x64xbf16>
  }
}
