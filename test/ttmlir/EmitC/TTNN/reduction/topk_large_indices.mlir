// REQUIRES: Blackhole
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %t.cpp %t2.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// The ttcore.composite "topk_large_indices" is promoted to
// ttnn.topk_large_indices by TTNNResolveComposites and then emitted to C++ as
// ttnn::experimental::topk_large_indices.

module {
  func.func @topk_large_indices(%input: tensor<1x512xbf16>) -> tensor<1x512xui32> {
    // CHECK: ttnn::experimental::topk_large_indices
    %0 = "ttcore.composite"(%input) <{composite_name = "topk_large_indices", decomposition = @decomp, composite_attributes = {k = 512 : ui32}}> : (tensor<1x512xbf16>) -> tensor<1x512xui32>
    return %0 : tensor<1x512xui32>
  }
  func.func private @decomp(%input: tensor<1x512xbf16>) -> tensor<1x512xui32> {
    %values, %indices = "ttir.topk"(%input) <{k = 512 : i32, dim = -1 : i32, largest = true, sorted = true}> : (tensor<1x512xbf16>) -> (tensor<1x512xbf16>, tensor<1x512xui32>)
    return %indices : tensor<1x512xui32>
  }
}
