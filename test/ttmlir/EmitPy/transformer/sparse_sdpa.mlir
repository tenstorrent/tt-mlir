// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="mock-system-desc-arch=blackhole composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// The ttcore.composite "sparse_sdpa" is promoted to ttnn.sparse_sdpa by
// TTNNResolveComposites and then emitted to Python as
// ttnn.transformer.sparse_sdpa.

func.func @sparse_sdpa(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
  // CHECK-LABEL: def sparse_sdpa
  // CHECK: ttnn.transformer.sparse_sdpa({{[a-z_0-9]+}}, {{[a-z_0-9]+}}, {{[a-z_0-9]+}}, 32, scale=None, k_chunk_size=32)
  %0 = "ttcore.composite"(%q, %kv, %idx) <{composite_name = "sparse_sdpa", decomposition = @decomp, composite_attributes = {v_dim = 32 : ui32, k_chunk_size = 32 : ui32}}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
  return %0 : tensor<1x32x32x32xbf16>
}
func.func private @decomp(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
  %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x32xbf16>
  return %0 : tensor<1x32x32x32xbf16>
}

// -----

// An explicit scale is forwarded as the `scale` keyword.
func.func @sparse_sdpa_scaled(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
  // CHECK-LABEL: def sparse_sdpa_scaled
  // CHECK: ttnn.transformer.sparse_sdpa({{[a-z_0-9]+}}, {{[a-z_0-9]+}}, {{[a-z_0-9]+}}, 32, scale=0.125, k_chunk_size=32)
  %0 = "ttcore.composite"(%q, %kv, %idx) <{composite_name = "sparse_sdpa", decomposition = @decomp_scaled, composite_attributes = {v_dim = 32 : ui32, k_chunk_size = 32 : ui32, scale = 1.250000e-01 : f32}}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
  return %0 : tensor<1x32x32x32xbf16>
}
func.func private @decomp_scaled(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
  %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x32xbf16>
  return %0 : tensor<1x32x32x32xbf16>
}
