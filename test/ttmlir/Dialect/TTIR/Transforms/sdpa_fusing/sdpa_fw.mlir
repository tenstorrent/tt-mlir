// RUN: ttmlir-opt --split-input-file --canonicalize --ttir-fusing %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file --canonicalize --ttir-fusing --ttnn-resolve-composites="composite-resolution=inline" --canonicalize --symbol-dce %s | FileCheck %s --check-prefix=INLINE
// RUN: ttmlir-opt --split-input-file --canonicalize --ttnn-resolve-composites="composite-resolution=inline" --ttir-fusing --symbol-dce %s | FileCheck %s --check-prefix=INLINE
// RUN: ttmlir-opt --split-input-file --canonicalize --ttir-fusing --ttir-to-ttnn-backend-pipeline="optimization-level=0 composite-resolution=force-promote mock-system-desc-arch=wormhole_b0" %s | FileCheck %s --check-prefix=PIPELINE

// The shape-only decomposition uses every input, so inlining must reconstruct
// exactly the original computation. A second call still needs the original
// decomposition's expanded signature.
module {
  // CHECK-LABEL: func.func @shared_decomposition(
  // CHECK-NOT: ttir.repeat_interleave
  // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2, %arg3)
  // CHECK-SAME: decomposition = @decomp_gqa
  // CHECK: "ttcore.composite"(%arg0, %arg0, %arg0, %arg3)
  // CHECK-SAME: decomposition = @decomp}>
  // PIPELINE-LABEL: func.func @shared_decomposition(
  // PIPELINE-NOT: ttnn.repeat_interleave
  // PIPELINE: "ttnn.sdpa_fw"
  // PIPELINE-SAME: tensor<1x2x64x64xbf16, {{[^>]+}}>, tensor<1x2x64x64xbf16,
  // PIPELINE: "ttnn.sdpa_fw"
  // INLINE-LABEL: func.func @shared_decomposition(
  // INLINE: %[[K:.*]] = "ttir.repeat_interleave"(%arg1)
  // INLINE: %[[V:.*]] = "ttir.repeat_interleave"(%arg2)
  // INLINE: "ttir.add"(%arg0, %[[K]])
  // INLINE: "ttir.multiply"(%{{.*}}, %[[V]])
  // INLINE-NOT: ttcore.composite
  // INLINE: return
  func.func @shared_decomposition(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x2x64x64xbf16>, %v: tensor<1x2x64x64xbf16>, %mask: tensor<1x1x64x64xbf16>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) {
    %ke = "ttir.repeat_interleave"(%k) <{dim = -3 : si32, repeats = 4 : ui32}> : (tensor<1x2x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %ve = "ttir.repeat_interleave"(%v) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x2x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %out, %lse = "ttcore.composite"(%q, %ke, %ve, %mask) <{composite_name = "sdpa_fw", decomposition = @decomp, composite_attributes = {mask_type = #ttcore.attention_mask_type<arbitrary>, dropout_probability = 0.000000e+00 : f32, return_intermediates = true}}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
    %other, %other_lse = "ttcore.composite"(%q, %q, %q, %mask) <{composite_name = "sdpa_fw", decomposition = @decomp, composite_attributes = {mask_type = #ttcore.attention_mask_type<arbitrary>, dropout_probability = 0.000000e+00 : f32, return_intermediates = true}}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
    return %out, %lse, %other, %other_lse : tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>
  }

  // CHECK-LABEL: func.func private @decomp(
  // CHECK-SAME: %arg1: tensor<1x8x64x64xbf16>, %arg2: tensor<1x8x64x64xbf16>
  // CHECK-NOT: ttir.repeat_interleave
  // CHECK: return
  func.func private @decomp(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x8x64x64xbf16>, %v: tensor<1x8x64x64xbf16>, %mask: tensor<1x1x64x64xbf16>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) {
    %a = "ttir.add"(%q, %k) : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %b = "ttir.multiply"(%a, %v) : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %out = "ttir.add"(%b, %mask) : (tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %lse = "ttir.empty"() : () -> tensor<1x8x64x32xf32>
    return %out, %lse : tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>
  }
  // CHECK-LABEL: func.func private @decomp_gqa(
  // CHECK-SAME: %arg1: tensor<1x2x64x64xbf16>, %arg2: tensor<1x2x64x64xbf16>
  // CHECK: %[[K:.*]] = "ttir.repeat_interleave"(%arg1)
  // CHECK-SAME: dim = 1 : si32, repeats = 4 : ui32
  // CHECK: %[[V:.*]] = "ttir.repeat_interleave"(%arg2)
  // CHECK: "ttir.add"(%arg0, %[[K]])
  // CHECK: "ttir.multiply"(%{{.*}}, %[[V]])
  // CHECK: "ttir.add"(%{{.*}}, %arg3)
  // CHECK: return
}

// -----

// HF-style repeat_kv is normalized by the same fusing pass before the shared
// head-expansion pattern runs. Casts stay outside on the smaller tensors.
module {
  // CHECK-LABEL: func.func @reshape_cast(
  // CHECK-NOT: ttir.repeat_interleave
  // CHECK: %[[K:.*]] = "ttir.typecast"(%arg1)
  // CHECK-SAME: (tensor<1x2x64x64xf32>) -> tensor<1x2x64x64xbf16>
  // CHECK: %[[V:.*]] = "ttir.typecast"(%arg2)
  // CHECK-SAME: (tensor<1x2x64x64xf32>) -> tensor<1x2x64x64xbf16>
  // CHECK: "ttcore.composite"(%arg0, %[[K]], %[[V]])
  // CHECK-SAME: decomposition = @decomp_gqa
  // PIPELINE-LABEL: func.func @reshape_cast(
  // PIPELINE-NOT: ttnn.repeat_interleave
  // PIPELINE: "ttnn.sdpa_fw"
  // PIPELINE-SAME: tensor<1x2x64x64xbf16, {{[^>]+}}>, tensor<1x2x64x64xbf16,
  // INLINE-LABEL: func.func @reshape_cast(
  // INLINE: "ttir.repeat_interleave"
  // INLINE: "ttir.repeat_interleave"
  // INLINE: "ttir.add"
  // INLINE: "ttir.multiply"
  // INLINE: return
  func.func @reshape_cast(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x2x64x64xf32>, %v: tensor<1x2x64x64xf32>) -> tensor<1x8x64x64xbf16> {
    %ku = "ttir.reshape"(%k) <{shape = [1 : i32, 2 : i32, 1 : i32, 64 : i32, 64 : i32]}> : (tensor<1x2x64x64xf32>) -> tensor<1x2x1x64x64xf32>
    %kb = "ttir.broadcast"(%ku) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x2x1x64x64xf32>) -> tensor<1x2x4x64x64xf32>
    %ke = "ttir.reshape"(%kb) <{shape = [1 : i32, 8 : i32, 64 : i32, 64 : i32]}> : (tensor<1x2x4x64x64xf32>) -> tensor<1x8x64x64xf32>
    %vu = "ttir.unsqueeze"(%v) <{dim = 2 : si32}> : (tensor<1x2x64x64xf32>) -> tensor<1x2x1x64x64xf32>
    %vb = "ttir.broadcast"(%vu) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x2x1x64x64xf32>) -> tensor<1x2x4x64x64xf32>
    %ve = "ttir.reshape"(%vb) <{shape = [1 : i32, 8 : i32, 64 : i32, 64 : i32]}> : (tensor<1x2x4x64x64xf32>) -> tensor<1x8x64x64xf32>
    %kc = "ttir.typecast"(%ke) : (tensor<1x8x64x64xf32>) -> tensor<1x8x64x64xbf16>
    %vc = "ttir.typecast"(%ve) : (tensor<1x8x64x64xf32>) -> tensor<1x8x64x64xbf16>
    %out = "ttcore.composite"(%q, %kc, %vc) <{composite_name = "sdpa_fw", decomposition = @decomp, composite_attributes = {mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32, return_intermediates = false}}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %out : tensor<1x8x64x64xbf16>
  }

  func.func private @decomp(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x8x64x64xbf16>, %v: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    %a = "ttir.add"(%q, %k) : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %out = "ttir.multiply"(%a, %v) : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %out : tensor<1x8x64x64xbf16>
  }
  // CHECK-LABEL: func.func private @decomp_gqa(
  // CHECK-SAME: %arg1: tensor<1x2x64x64xbf16>, %arg2: tensor<1x2x64x64xbf16>
  // CHECK-NOT: ttir.typecast
  // CHECK: "ttir.repeat_interleave"(%arg1)
  // CHECK: "ttir.repeat_interleave"(%arg2)
  // CHECK: return
}

// -----

// sdpa_fw allows a different value head dimension. Both calls fuse, exercising
// independent specializations and unique names for a shared decomposition.
module {
  // CHECK-LABEL: func.func @value_head_dim(
  // CHECK-NOT: ttir.repeat_interleave
  // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
  // CHECK-SAME: decomposition = @decomp_gqa
  // CHECK-SAME: tensor<1x2x64x64xbf16>, tensor<1x2x64x32xbf16>
  // CHECK: "ttcore.composite"(%arg0, %arg3, %arg4)
  // CHECK-SAME: decomposition = @decomp_gqa_0
  // PIPELINE-LABEL: func.func @value_head_dim(
  // PIPELINE-NOT: ttnn.repeat_interleave
  // PIPELINE: "ttnn.sdpa_fw"
  // PIPELINE-SAME: tensor<1x2x64x64xbf16, {{[^>]+}}>, tensor<1x2x64x32xbf16,
  // PIPELINE: "ttnn.sdpa_fw"
  // PIPELINE-SAME: tensor<1x4x64x64xbf16, {{[^>]+}}>, tensor<1x4x64x32xbf16,
  func.func @value_head_dim(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x2x64x64xbf16>, %v: tensor<1x2x64x32xbf16>, %k2: tensor<1x4x64x64xbf16>, %v2: tensor<1x4x64x32xbf16>) -> (tensor<1x8x64x32xbf16>, tensor<1x8x64x32xbf16>) {
    %ke = "ttir.repeat_interleave"(%k) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x2x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %ve = "ttir.repeat_interleave"(%v) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x2x64x32xbf16>) -> tensor<1x8x64x32xbf16>
    %ke2 = "ttir.repeat_interleave"(%k2) <{dim = 1 : si32, repeats = 2 : ui32}> : (tensor<1x4x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %ve2 = "ttir.repeat_interleave"(%v2) <{dim = 1 : si32, repeats = 2 : ui32}> : (tensor<1x4x64x32xbf16>) -> tensor<1x8x64x32xbf16>
    %out = "ttcore.composite"(%q, %ke, %ve) <{composite_name = "sdpa_fw", decomposition = @decomp, composite_attributes = {mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32, return_intermediates = false}}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xbf16>) -> tensor<1x8x64x32xbf16>
    %out2 = "ttcore.composite"(%q, %ke2, %ve2) <{composite_name = "sdpa_fw", decomposition = @decomp, composite_attributes = {mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32, return_intermediates = false}}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xbf16>) -> tensor<1x8x64x32xbf16>
    return %out, %out2 : tensor<1x8x64x32xbf16>, tensor<1x8x64x32xbf16>
  }

  func.func private @decomp(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x8x64x64xbf16>, %v: tensor<1x8x64x32xbf16>) -> tensor<1x8x64x32xbf16> {
    %a = "ttir.add"(%q, %k) : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %out = "ttir.matmul"(%a, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xbf16>) -> tensor<1x8x64x32xbf16>
    return %out : tensor<1x8x64x32xbf16>
  }
}

// -----

module {
  // CHECK-LABEL: func.func @nonmatching_expansions(
  // CHECK: "ttir.repeat_interleave"
  // CHECK: "ttir.repeat_interleave"
  // CHECK: "ttir.repeat_interleave"
  // CHECK: "ttir.repeat_interleave"
  // CHECK-COUNT-4: decomposition = @decomp}>
  // CHECK-NOT: @decomp_gqa
  func.func @nonmatching_expansions(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x2x64x64xbf16>, %v: tensor<1x4x64x64xbf16>, %s: tensor<1x8x16x64xbf16>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    %ke = "ttir.repeat_interleave"(%k) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x2x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %ve = "ttir.repeat_interleave"(%v) <{dim = 1 : si32, repeats = 2 : ui32}> : (tensor<1x4x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %se = "ttir.repeat_interleave"(%s) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x8x16x64xbf16>) -> tensor<1x8x64x64xbf16>
    %wrong_dim = "ttir.repeat_interleave"(%s) <{dim = -2 : si32, repeats = 4 : ui32}> : (tensor<1x8x16x64xbf16>) -> tensor<1x8x64x64xbf16>
    // Only K expanded, mismatched repeat counts, wrong dimension, and unrelated composite.
    %a = "ttcore.composite"(%q, %ke, %q) <{composite_name = "sdpa_fw", decomposition = @decomp, composite_attributes = {mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32, return_intermediates = false}}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %b = "ttcore.composite"(%q, %ke, %ve) <{composite_name = "sdpa_fw", decomposition = @decomp, composite_attributes = {mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32, return_intermediates = false}}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %c = "ttcore.composite"(%q, %se, %wrong_dim) <{composite_name = "sdpa_fw", decomposition = @decomp, composite_attributes = {mask_type = #ttcore.attention_mask_type<causal>, dropout_probability = 0.000000e+00 : f32, return_intermediates = false}}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %d = "ttcore.composite"(%q, %ke, %ke) <{composite_name = "other", decomposition = @decomp}> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %a, %b, %c, %d : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }

  func.func private @decomp(%q: tensor<1x8x64x64xbf16>, %k: tensor<1x8x64x64xbf16>, %v: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    %a = "ttir.add"(%q, %k) : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %out = "ttir.multiply"(%a, %v) : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %out : tensor<1x8x64x64xbf16>
  }
}
