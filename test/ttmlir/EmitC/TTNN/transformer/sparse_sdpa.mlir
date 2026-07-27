// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="mock-system-desc-arch=blackhole enable-cpu-hoisted-const-eval=false composite-resolution=force-promote" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %t.cpp %t2.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// The ttcore.composite "sparse_sdpa" is promoted to ttnn.sparse_sdpa by
// TTNNResolveComposites and then emitted to C++ as
// ttnn::transformer::sparse_sdpa.

module {
  func.func @sparse_sdpa(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // CHECK: ttnn::transformer::sparse_sdpa
    %0 = "ttcore.composite"(%q, %kv, %idx) <{composite_name = "sparse_sdpa", decomposition = @decomp, composite_attributes = {v_dim = 32 : ui32, k_chunk_size = 32 : ui32}}> : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
  func.func private @decomp(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}
