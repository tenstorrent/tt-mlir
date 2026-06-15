// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir

// The ttcore.composite "flash_mla_prefill" is promoted to ttnn.flash_mla_prefill
// by TTNNResolveComposites and then emitted to Python.

func.func @flash_mla_prefill_causal_no_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
  // CHECK: ttnn.transformer.flash_mla_prefill
  %0 = "ttcore.composite"(%query, %key) <{composite_name = "flash_mla_prefill", decomposition = @decomp_no_value, composite_attributes = {head_dim_v = 64 : ui32, is_causal = true, has_value = false, has_attention_mask = false}}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
  return %0 : tensor<1x16x32x64xbf16>
}
func.func private @decomp_no_value(%q: tensor<1x16x32x128xbf16>, %k: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
  %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x32x128xbf16>) -> tensor<1x16x32x64xbf16>
  return %0 : tensor<1x16x32x64xbf16>
}

func.func @flash_mla_prefill_causal_with_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
  // CHECK: ttnn.transformer.flash_mla_prefill
  %0 = "ttcore.composite"(%query, %key, %value) <{composite_name = "flash_mla_prefill", decomposition = @decomp_with_value, composite_attributes = {head_dim_v = 64 : ui32, is_causal = true, has_value = true, has_attention_mask = false}}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
  return %0 : tensor<1x16x32x64xbf16>
}
func.func private @decomp_with_value(%q: tensor<1x16x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %v: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
  %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x32x128xbf16>) -> tensor<1x16x32x64xbf16>
  return %0 : tensor<1x16x32x64xbf16>
}

func.func @flash_mla_prefill_with_mask(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
  // CHECK: ttnn.transformer.flash_mla_prefill
  %0 = "ttcore.composite"(%query, %key, %mask) <{composite_name = "flash_mla_prefill", decomposition = @decomp_with_mask, composite_attributes = {head_dim_v = 64 : ui32, is_causal = false, has_value = false, has_attention_mask = true}}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16>
  return %0 : tensor<1x16x32x64xbf16>
}
func.func private @decomp_with_mask(%q: tensor<1x16x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %m: tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
  %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x32x128xbf16>) -> tensor<1x16x32x64xbf16>
  return %0 : tensor<1x16x32x64xbf16>
}

func.func @flash_mla_prefill_value_mask_scale(%query: tensor<2x8x64x128xbf16>, %key: tensor<2x1x64x128xbf16>, %value: tensor<2x1x64x96xbf16>, %mask: tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16> {
  // CHECK: ttnn.transformer.flash_mla_prefill
  %0 = "ttcore.composite"(%query, %key, %value, %mask) <{composite_name = "flash_mla_prefill", decomposition = @decomp_value_mask_scale, composite_attributes = {head_dim_v = 96 : ui32, is_causal = false, scale = 1.250000e-01 : f32, has_value = true, has_attention_mask = true}}> : (tensor<2x8x64x128xbf16>, tensor<2x1x64x128xbf16>, tensor<2x1x64x96xbf16>, tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16>
  return %0 : tensor<2x8x64x96xbf16>
}
func.func private @decomp_value_mask_scale(%q: tensor<2x8x64x128xbf16>, %k: tensor<2x1x64x128xbf16>, %v: tensor<2x1x64x96xbf16>, %m: tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16> {
  %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 8 : i32, 64 : i32, 96 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x8x64x128xbf16>) -> tensor<2x8x64x96xbf16>
  return %0 : tensor<2x8x64x96xbf16>
}
