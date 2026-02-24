// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t_fb.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp
module @sdpa {
    func.func public @sdpa_causal(%arg0: tensor<1x1x12x64xbf16>, %arg1: tensor<128x12x32x64xbf16>, %arg2: tensor<128x12x32x64xbf16>, %arg3: tensor<1x4xi32>, %arg4: tensor<1xi32>) -> tensor<1x1x12x64xbf16> {
        %0 = ttir.empty() : tensor<1x1x12x64xbf16>
        %1 = "ttir.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %0, %arg4) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 0>}> : (tensor<1x1x12x64xbf16>, tensor<128x12x32x64xbf16>, tensor<128x12x32x64xbf16>, tensor<1x4xi32>, tensor<1x1x12x64xbf16>, tensor<1xi32>) -> tensor<1x1x12x64xbf16>
        return %1 : tensor<1x1x12x64xbf16>
    }
}
