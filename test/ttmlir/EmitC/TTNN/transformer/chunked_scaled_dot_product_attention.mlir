// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path% composite-resolution=force-promote" %s > %t.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline %t.mlir > %t_rt.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t_rt.mlir > %basename_t.ttnn
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp
// RUN: FileCheck %s --input-file=%basename_t.cpp

// The ttcore.composite "chunked_scaled_dot_product_attention" is promoted to
// ttnn.chunked_scaled_dot_product_attention by TTNNResolveComposites and then
// emitted to C++.
module @chunked_sdpa {
    func.func public @chunked_sdpa(%arg0: tensor<1x12x64x64xbf16>, %arg1: tensor<128x12x32x64xbf16>, %arg2: tensor<128x12x32x64xbf16>, %arg3: tensor<1x8xi32>, %arg4: tensor<1xi32>) -> tensor<1x12x64x64xbf16> {
        // CHECK: ttnn::transformer::chunked_scaled_dot_product_attention(
        %0 = "ttcore.composite"(%arg0, %arg1, %arg2, %arg3, %arg4) <{composite_name = "chunked_scaled_dot_product_attention", decomposition = @chunked_scaled_dot_product_attention_decomp, composite_attributes = {scale = 1.250000e-01 : f32}}> : (tensor<1x12x64x64xbf16>, tensor<128x12x32x64xbf16>, tensor<128x12x32x64xbf16>, tensor<1x8xi32>, tensor<1xi32>) -> tensor<1x12x64x64xbf16>
        return %0 : tensor<1x12x64x64xbf16>
    }
    func.func private @chunked_scaled_dot_product_attention_decomp(%query: tensor<1x12x64x64xbf16>, %key: tensor<128x12x32x64xbf16>, %value: tensor<128x12x32x64xbf16>, %page_table: tensor<1x8xi32>, %chunk_start_idx: tensor<1xi32>) -> tensor<1x12x64x64xbf16> {
        return %query : tensor<1x12x64x64xbf16>
    }
}
