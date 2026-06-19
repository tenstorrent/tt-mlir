// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" %s > %t.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline %t.mlir > %t_rt.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t_rt.mlir > %basename_t.ttnn
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp
// RUN: FileCheck %s --input-file=%basename_t.cpp
module @chunked_sdpa {
    func.func public @chunked_sdpa(%arg0: tensor<1x12x64x64xbf16>, %arg1: tensor<128x12x32x64xbf16>, %arg2: tensor<128x12x32x64xbf16>, %arg3: tensor<1x8xi32>, %arg4: tensor<1xi32>) -> tensor<1x12x64x64xbf16> {
        // CHECK: ttnn::transformer::chunked_scaled_dot_product_attention(
        %0 = ttir.empty() : tensor<1x12x64x64xbf16>
        %1 = "ttir.chunked_scaled_dot_product_attention"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{scale = 1.250000e-01 : f32}> : (tensor<1x12x64x64xbf16>, tensor<128x12x32x64xbf16>, tensor<128x12x32x64xbf16>, tensor<1x8xi32>, tensor<1xi32>, tensor<1x12x64x64xbf16>) -> tensor<1x12x64x64xbf16>
        return %1 : tensor<1x12x64x64xbf16>
    }
}
