// REQUIRES: opmodel
// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

// The ttcore.composite "flash_mla_prefill" is promoted to ttnn.flash_mla_prefill
// by TTNNResolveComposites and then emitted to C++.

module {
    func.func @flash_mla_prefill_causal_no_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
        // CHECK: "ttnn.flash_mla_prefill"
        %0 = "ttcore.composite"(%query, %key) <{composite_name = "flash_mla_prefill", decomposition = @decomp_no_value, composite_attributes = {head_dim_v = 64 : ui32, is_causal = true, has_value = false, has_attention_mask = false}}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
        return %0 : tensor<1x16x32x64xbf16>
    }
    func.func private @decomp_no_value(%q: tensor<1x16x32x128xbf16>, %k: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
        %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x32x128xbf16>) -> tensor<1x16x32x64xbf16>
        return %0 : tensor<1x16x32x64xbf16>
    }

    func.func @flash_mla_prefill_with_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
        // CHECK: "ttnn.flash_mla_prefill"
        %0 = "ttcore.composite"(%query, %key, %value) <{composite_name = "flash_mla_prefill", decomposition = @decomp_with_value, composite_attributes = {head_dim_v = 64 : ui32, is_causal = true, has_value = true, has_attention_mask = false}}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
        return %0 : tensor<1x16x32x64xbf16>
    }
    func.func private @decomp_with_value(%q: tensor<1x16x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %v: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
        %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x32x128xbf16>) -> tensor<1x16x32x64xbf16>
        return %0 : tensor<1x16x32x64xbf16>
    }
}
