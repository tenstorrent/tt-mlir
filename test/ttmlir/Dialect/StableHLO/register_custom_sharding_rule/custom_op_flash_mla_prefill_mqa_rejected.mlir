// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline %s 2>&1 | FileCheck %s

// MQA -- a single shared KV head (kvHeads == 1) -- is the MLA *decode* form,
// not prefill (prefill up-projects the latent to per-head K/V, i.e. MHA).
// Feeding the decode/latent form to the prefill op makes the custom sharding
// rule reject it with a clear error (and decline to provide a rule) rather
// than letting it surface later as an illegal Shardy all_slice on the size-1
// KV head. Uses the DeepSeek V3.1 latent dims (dh_qk = 512 + 64,
// head_dim_v = 512) to mimic a decode tensor wrongly routed to prefill.
// CHECK: error: flash_mla_prefill (MLA prefill) expects MHA inputs but got num_kv_heads == 1 (MQA)
module @FlashMlaPrefill_MQA_Rejected attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%query: tensor<1x128x2048x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x1x2048x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}) -> tensor<1x128x2048x512xbf16> {
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "512", is_causal = "True", has_value = "False", has_attention_mask = "False"}} : (tensor<1x128x2048x576xbf16>, tensor<1x1x2048x576xbf16>) -> tensor<1x128x2048x512xbf16>
    return %0 : tensor<1x128x2048x512xbf16>
  }
}
