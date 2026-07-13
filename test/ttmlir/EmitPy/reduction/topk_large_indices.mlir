// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="mock-system-desc-arch=blackhole composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// The ttcore.composite "topk_large_indices" is promoted to
// ttnn.topk_large_indices by TTNNResolveComposites and then emitted to Python
// as ttnn.experimental.topk_large_indices.

func.func @topk_large_indices(%input: tensor<1x512xbf16>) -> tensor<1x512xui32> {
  // CHECK-LABEL: def topk_large_indices
  // CHECK: ttnn.experimental.topk_large_indices({{[a-z_0-9]+}}, k=512)
  %0 = "ttcore.composite"(%input) <{composite_name = "topk_large_indices", decomposition = @decomp, composite_attributes = {k = 512 : ui32}}> : (tensor<1x512xbf16>) -> tensor<1x512xui32>
  return %0 : tensor<1x512xui32>
}
func.func private @decomp(%input: tensor<1x512xbf16>) -> tensor<1x512xui32> {
  %values, %indices = "ttir.topk"(%input) <{k = 512 : i32, dim = -1 : i32, largest = true, sorted = true}> : (tensor<1x512xbf16>) -> (tensor<1x512xbf16>, tensor<1x512xui32>)
  return %indices : tensor<1x512xui32>
}
