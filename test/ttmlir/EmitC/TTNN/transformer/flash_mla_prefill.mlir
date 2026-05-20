// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" %s > %t.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline %t.mlir > %t_rt.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t_rt.mlir > %basename_t.ttnn
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

module {
    func.func @flash_mla_prefill_causal_no_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
        // CHECK: "ttnn.flash_mla_prefill"
        %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
        return %0 : tensor<1x16x32x64xbf16>
    }

    func.func @flash_mla_prefill_with_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
        // CHECK: "ttnn.flash_mla_prefill"
        %0 = "ttir.flash_mla_prefill"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
        return %0 : tensor<1x16x32x64xbf16>
    }
}
