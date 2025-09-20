// -----// IR Dump After ConvertStableHLOToTTIR (convert-stablehlo-to-ttir) ('builtin.module' operation: @SyncTensorsGraph.7624) //----- //
module @SyncTensorsGraph.7624 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func @main(%arg0: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_v_proj_weight"}, %arg1: tensor<f32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "auto_annotated_const_0"}, %arg2: tensor<1x640xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"}, %arg3: tensor<128256x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_embed_tokens_weight"}, %arg4: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___input_layernorm_weight"}, %arg5: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__1___self_attn_v_proj_weight"}, %arg6: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___mlp_down_proj_weight"}, %arg7: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___mlp_up_proj_weight"}, %arg8: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_o_proj_weight"}, %arg9: tensor<bf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "auto_annotated_const_1"}, %arg10: tensor<1x640xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_1"}, %arg11: tensor<f32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "auto_annotated_const_2"}, %arg12: tensor<64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_rotary_emb_inv_freq"}, %arg13: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_k_proj_weight"}, %arg14: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_q_proj_weight"}, %arg15: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___post_attention_layernorm_weight"}, %arg16: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___mlp_gate_proj_weight"}, %arg17: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__1___input_layernorm_weight"}, %arg18: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__2___self_attn_v_proj_weight"}, %arg19: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__1___mlp_down_proj_weight"}, %arg20: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__1___mlp_up_proj_weight"}, %arg21: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__1___self_attn_o_proj_weight"}, %arg22: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__1___self_attn_k_proj_weight"}, %arg23: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__1___self_attn_q_proj_weight"}, %arg24: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__1___post_attention_layernorm_weight"}, %arg25: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__1___mlp_gate_proj_weight"}, %arg26: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__2___input_layernorm_weight"}, %arg27: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__3___self_attn_v_proj_weight"}, %arg28: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__2___mlp_down_proj_weight"}, %arg29: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__2___mlp_up_proj_weight"}, %arg30: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__2___self_attn_o_proj_weight"}, %arg31: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__2___self_attn_k_proj_weight"}, %arg32: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__2___self_attn_q_proj_weight"}, %arg33: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__2___post_attention_layernorm_weight"}, %arg34: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__2___mlp_gate_proj_weight"}, %arg35: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__3___input_layernorm_weight"}, %arg36: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__4___self_attn_v_proj_weight"}, %arg37: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__3___mlp_down_proj_weight"}, %arg38: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__3___mlp_up_proj_weight"}, %arg39: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__3___self_attn_o_proj_weight"}, %arg40: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__3___self_attn_k_proj_weight"}, %arg41: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__3___self_attn_q_proj_weight"}, %arg42: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__3___post_attention_layernorm_weight"}, %arg43: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__3___mlp_gate_proj_weight"}, %arg44: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__4___input_layernorm_weight"}, %arg45: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__5___self_attn_v_proj_weight"}, %arg46: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__4___mlp_down_proj_weight"}, %arg47: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__4___mlp_up_proj_weight"}, %arg48: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__4___self_attn_o_proj_weight"}, %arg49: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__4___self_attn_k_proj_weight"}, %arg50: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__4___self_attn_q_proj_weight"}, %arg51: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__4___post_attention_layernorm_weight"}, %arg52: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__4___mlp_gate_proj_weight"}, %arg53: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__5___input_layernorm_weight"}, %arg54: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__6___self_attn_v_proj_weight"}, %arg55: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__5___mlp_down_proj_weight"}, %arg56: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__5___mlp_up_proj_weight"}, %arg57: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__5___self_attn_o_proj_weight"}, %arg58: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__5___self_attn_k_proj_weight"}, %arg59: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__5___self_attn_q_proj_weight"}, %arg60: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__5___post_attention_layernorm_weight"}, %arg61: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__5___mlp_gate_proj_weight"}, %arg62: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__6___input_layernorm_weight"}, %arg63: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__7___self_attn_v_proj_weight"}, %arg64: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__6___mlp_down_proj_weight"}, %arg65: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__6___mlp_up_proj_weight"}, %arg66: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__6___self_attn_o_proj_weight"}, %arg67: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__6___self_attn_k_proj_weight"}, %arg68: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__6___self_attn_q_proj_weight"}, %arg69: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__6___post_attention_layernorm_weight"}, %arg70: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__6___mlp_gate_proj_weight"}, %arg71: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__7___input_layernorm_weight"}, %arg72: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__8___self_attn_v_proj_weight"}, %arg73: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__7___mlp_down_proj_weight"}, %arg74: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__7___mlp_up_proj_weight"}, %arg75: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__7___self_attn_o_proj_weight"}, %arg76: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__7___self_attn_k_proj_weight"}, %arg77: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__7___self_attn_q_proj_weight"}, %arg78: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__7___post_attention_layernorm_weight"}, %arg79: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__7___mlp_gate_proj_weight"}, %arg80: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__8___input_layernorm_weight"}, %arg81: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__9___self_attn_v_proj_weight"}, %arg82: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__8___mlp_down_proj_weight"}, %arg83: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__8___mlp_up_proj_weight"}, %arg84: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__8___self_attn_o_proj_weight"}, %arg85: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__8___self_attn_k_proj_weight"}, %arg86: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__8___self_attn_q_proj_weight"}, %arg87: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__8___post_attention_layernorm_weight"}, %arg88: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__8___mlp_gate_proj_weight"}, %arg89: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__9___input_layernorm_weight"}, %arg90: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__10___self_attn_v_proj_weight"}, %arg91: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__9___mlp_down_proj_weight"}, %arg92: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__9___mlp_up_proj_weight"}, %arg93: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__9___self_attn_o_proj_weight"}, %arg94: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__9___self_attn_k_proj_weight"}, %arg95: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__9___self_attn_q_proj_weight"}, %arg96: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__9___post_attention_layernorm_weight"}, %arg97: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__9___mlp_gate_proj_weight"}, %arg98: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__10___input_layernorm_weight"}, %arg99: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__11___self_attn_v_proj_weight"}, %arg100: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__10___mlp_down_proj_weight"}, %arg101: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__10___mlp_up_proj_weight"}, %arg102: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__10___self_attn_o_proj_weight"}, %arg103: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__10___self_attn_k_proj_weight"}, %arg104: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__10___self_attn_q_proj_weight"}, %arg105: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__10___post_attention_layernorm_weight"}, %arg106: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__10___mlp_gate_proj_weight"}, %arg107: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__11___input_layernorm_weight"}, %arg108: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__12___self_attn_v_proj_weight"}, %arg109: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__11___mlp_down_proj_weight"}, %arg110: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__11___mlp_up_proj_weight"}, %arg111: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__11___self_attn_o_proj_weight"}, %arg112: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__11___self_attn_k_proj_weight"}, %arg113: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__11___self_attn_q_proj_weight"}, %arg114: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__11___post_attention_layernorm_weight"}, %arg115: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__11___mlp_gate_proj_weight"}, %arg116: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__12___input_layernorm_weight"}, %arg117: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__13___self_attn_v_proj_weight"}, %arg118: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__12___mlp_down_proj_weight"}, %arg119: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__12___mlp_up_proj_weight"}, %arg120: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__12___self_attn_o_proj_weight"}, %arg121: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__12___self_attn_k_proj_weight"}, %arg122: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__12___self_attn_q_proj_weight"}, %arg123: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__12___post_attention_layernorm_weight"}, %arg124: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__12___mlp_gate_proj_weight"}, %arg125: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__13___input_layernorm_weight"}, %arg126: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__14___self_attn_v_proj_weight"}, %arg127: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__13___mlp_down_proj_weight"}, %arg128: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__13___mlp_up_proj_weight"}, %arg129: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__13___self_attn_o_proj_weight"}, %arg130: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__13___self_attn_k_proj_weight"}, %arg131: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__13___self_attn_q_proj_weight"}, %arg132: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__13___post_attention_layernorm_weight"}, %arg133: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__13___mlp_gate_proj_weight"}, %arg134: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__14___input_layernorm_weight"}, %arg135: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__15___self_attn_v_proj_weight"}, %arg136: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__14___mlp_down_proj_weight"}, %arg137: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__14___mlp_up_proj_weight"}, %arg138: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__14___self_attn_o_proj_weight"}, %arg139: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__14___self_attn_k_proj_weight"}, %arg140: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__14___self_attn_q_proj_weight"}, %arg141: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__14___post_attention_layernorm_weight"}, %arg142: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__14___mlp_gate_proj_weight"}, %arg143: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__15___input_layernorm_weight"}, %arg144: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__16___self_attn_v_proj_weight"}, %arg145: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__15___mlp_down_proj_weight"}, %arg146: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__15___mlp_up_proj_weight"}, %arg147: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__15___self_attn_o_proj_weight"}, %arg148: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__15___self_attn_k_proj_weight"}, %arg149: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__15___self_attn_q_proj_weight"}, %arg150: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__15___post_attention_layernorm_weight"}, %arg151: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__15___mlp_gate_proj_weight"}, %arg152: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__16___input_layernorm_weight"}, %arg153: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__17___self_attn_v_proj_weight"}, %arg154: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__16___mlp_down_proj_weight"}, %arg155: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__16___mlp_up_proj_weight"}, %arg156: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__16___self_attn_o_proj_weight"}, %arg157: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__16___self_attn_k_proj_weight"}, %arg158: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__16___self_attn_q_proj_weight"}, %arg159: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__16___post_attention_layernorm_weight"}, %arg160: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__16___mlp_gate_proj_weight"}, %arg161: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__17___input_layernorm_weight"}, %arg162: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__18___self_attn_v_proj_weight"}, %arg163: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__17___mlp_down_proj_weight"}, %arg164: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__17___mlp_up_proj_weight"}, %arg165: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__17___self_attn_o_proj_weight"}, %arg166: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__17___self_attn_k_proj_weight"}, %arg167: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__17___self_attn_q_proj_weight"}, %arg168: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__17___post_attention_layernorm_weight"}, %arg169: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__17___mlp_gate_proj_weight"}, %arg170: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__18___input_layernorm_weight"}, %arg171: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__19___self_attn_v_proj_weight"}, %arg172: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__18___mlp_down_proj_weight"}, %arg173: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__18___mlp_up_proj_weight"}, %arg174: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__18___self_attn_o_proj_weight"}, %arg175: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__18___self_attn_k_proj_weight"}, %arg176: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__18___self_attn_q_proj_weight"}, %arg177: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__18___post_attention_layernorm_weight"}, %arg178: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__18___mlp_gate_proj_weight"}, %arg179: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__19___input_layernorm_weight"}, %arg180: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__20___self_attn_v_proj_weight"}, %arg181: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__19___mlp_down_proj_weight"}, %arg182: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__19___mlp_up_proj_weight"}, %arg183: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__19___self_attn_o_proj_weight"}, %arg184: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__19___self_attn_k_proj_weight"}, %arg185: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__19___self_attn_q_proj_weight"}, %arg186: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__19___post_attention_layernorm_weight"}, %arg187: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__19___mlp_gate_proj_weight"}, %arg188: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__20___input_layernorm_weight"}, %arg189: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__21___self_attn_v_proj_weight"}, %arg190: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__20___mlp_down_proj_weight"}, %arg191: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__20___mlp_up_proj_weight"}, %arg192: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__20___self_attn_o_proj_weight"}, %arg193: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__20___self_attn_k_proj_weight"}, %arg194: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__20___self_attn_q_proj_weight"}, %arg195: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__20___post_attention_layernorm_weight"}, %arg196: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__20___mlp_gate_proj_weight"}, %arg197: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__21___input_layernorm_weight"}, %arg198: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__22___self_attn_v_proj_weight"}, %arg199: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__21___mlp_down_proj_weight"}, %arg200: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__21___mlp_up_proj_weight"}, %arg201: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__21___self_attn_o_proj_weight"}, %arg202: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__21___self_attn_k_proj_weight"}, %arg203: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__21___self_attn_q_proj_weight"}, %arg204: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__21___post_attention_layernorm_weight"}, %arg205: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__21___mlp_gate_proj_weight"}, %arg206: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__22___input_layernorm_weight"}, %arg207: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__23___self_attn_v_proj_weight"}, %arg208: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__22___mlp_down_proj_weight"}, %arg209: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__22___mlp_up_proj_weight"}, %arg210: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__22___self_attn_o_proj_weight"}, %arg211: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__22___self_attn_k_proj_weight"}, %arg212: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__22___self_attn_q_proj_weight"}, %arg213: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__22___post_attention_layernorm_weight"}, %arg214: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__22___mlp_gate_proj_weight"}, %arg215: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__23___input_layernorm_weight"}, %arg216: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__24___self_attn_v_proj_weight"}, %arg217: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__23___mlp_down_proj_weight"}, %arg218: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__23___mlp_up_proj_weight"}, %arg219: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__23___self_attn_o_proj_weight"}, %arg220: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__23___self_attn_k_proj_weight"}, %arg221: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__23___self_attn_q_proj_weight"}, %arg222: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__23___post_attention_layernorm_weight"}, %arg223: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__23___mlp_gate_proj_weight"}, %arg224: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__24___input_layernorm_weight"}, %arg225: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__25___self_attn_v_proj_weight"}, %arg226: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__24___mlp_down_proj_weight"}, %arg227: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__24___mlp_up_proj_weight"}, %arg228: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__24___self_attn_o_proj_weight"}, %arg229: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__24___self_attn_k_proj_weight"}, %arg230: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__24___self_attn_q_proj_weight"}, %arg231: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__24___post_attention_layernorm_weight"}, %arg232: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__24___mlp_gate_proj_weight"}, %arg233: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__25___input_layernorm_weight"}, %arg234: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__26___self_attn_v_proj_weight"}, %arg235: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__25___mlp_down_proj_weight"}, %arg236: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__25___mlp_up_proj_weight"}, %arg237: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__25___self_attn_o_proj_weight"}, %arg238: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__25___self_attn_k_proj_weight"}, %arg239: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__25___self_attn_q_proj_weight"}, %arg240: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__25___post_attention_layernorm_weight"}, %arg241: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__25___mlp_gate_proj_weight"}, %arg242: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__26___input_layernorm_weight"}, %arg243: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__27___self_attn_v_proj_weight"}, %arg244: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__26___mlp_down_proj_weight"}, %arg245: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__26___mlp_up_proj_weight"}, %arg246: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__26___self_attn_o_proj_weight"}, %arg247: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__26___self_attn_k_proj_weight"}, %arg248: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__26___self_attn_q_proj_weight"}, %arg249: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__26___post_attention_layernorm_weight"}, %arg250: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__26___mlp_gate_proj_weight"}, %arg251: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__27___input_layernorm_weight"}, %arg252: tensor<1024x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__27___self_attn_k_proj_weight"}, %arg253: tensor<128256x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___lm_head_weight"}, %arg254: tensor<3072x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__27___mlp_down_proj_weight"}, %arg255: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__27___mlp_up_proj_weight"}, %arg256: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__27___self_attn_o_proj_weight"}, %arg257: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__27___self_attn_q_proj_weight"}, %arg258: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__27___post_attention_layernorm_weight"}, %arg259: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__27___mlp_gate_proj_weight"}, %arg260: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_norm_weight"}) -> (tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x8x640x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x640x128256xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x1x640x640xbf16>}> : () -> tensor<1x1x640x640xbf16>
    %1 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<640x640xbf16>}> : () -> tensor<640x640xbf16>
    %2 = "ttir.constant"() <{value = dense<1> : tensor<640x640xi64>}> : () -> tensor<640x640xi64>
    %3 = "ttir.constant"() <{value = dense<"0x000000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000A0410000A8410000B0410000B8410000C0410000C8410000D0410000D8410000E0410000E8410000F0410000F84100000042000004420000084200000C4200001042000014420000184200001C4200002042000024420000284200002C4200003042000034420000384200003C4200004042000044420000484200004C4200005042000054420000584200005C4200006042000064420000684200006C4200007042000074420000784200007C42000080420000824200008442000086420000884200008A4200008C4200008E42000090420000924200009442000096420000984200009A4200009C4200009E420000A0420000A2420000A4420000A6420000A8420000AA420000AC420000AE420000B0420000B2420000B4420000B6420000B8420000BA420000BC420000BE420000C0420000C2420000C4420000C6420000C8420000CA420000CC420000CE420000D0420000D2420000D4420000D6420000D8420000DA420000DC420000DE420000E0420000E2420000E4420000E6420000E8420000EA420000EC420000EE420000F0420000F2420000F4420000F6420000F8420000FA420000FC420000FE420000004300000143000002430000034300000443000005430000064300000743000008430000094300000A4300000B4300000C4300000D4300000E4300000F430000104300001143000012430000134300001443000015430000164300001743000018430000194300001A4300001B4300001C4300001D4300001E4300001F430000204300002143000022430000234300002443000025430000264300002743000028430000294300002A4300002B4300002C4300002D4300002E4300002F430000304300003143000032430000334300003443000035430000364300003743000038430000394300003A4300003B4300003C4300003D4300003E4300003F430000404300004143000042430000434300004443000045430000464300004743000048430000494300004A4300004B4300004C4300004D4300004E4300004F430000504300005143000052430000534300005443000055430000564300005743000058430000594300005A4300005B4300005C4300005D4300005E4300005F430000604300006143000062430000634300006443000065430000664300006743000068430000694300006A4300006B4300006C4300006D4300006E4300006F430000704300007143000072430000734300007443000075430000764300007743000078430000794300007A4300007B4300007C4300007D4300007E4300007F43000080430080804300008143008081430000824300808243000083430080834300008443008084430000854300808543000086430080864300008743008087430000884300808843000089430080894300008A4300808A4300008B4300808B4300008C4300808C4300008D4300808D4300008E4300808E4300008F4300808F43000090430080904300009143008091430000924300809243000093430080934300009443008094430000954300809543000096430080964300009743008097430000984300809843000099430080994300009A4300809A4300009B4300809B4300009C4300809C4300009D4300809D4300009E4300809E4300009F4300809F430000A0430080A0430000A1430080A1430000A2430080A2430000A3430080A3430000A4430080A4430000A5430080A5430000A6430080A6430000A7430080A7430000A8430080A8430000A9430080A9430000AA430080AA430000AB430080AB430000AC430080AC430000AD430080AD430000AE430080AE430000AF430080AF430000B0430080B0430000B1430080B1430000B2430080B2430000B3430080B3430000B4430080B4430000B5430080B5430000B6430080B6430000B7430080B7430000B8430080B8430000B9430080B9430000BA430080BA430000BB430080BB430000BC430080BC430000BD430080BD430000BE430080BE430000BF430080BF430000C0430080C0430000C1430080C1430000C2430080C2430000C3430080C3430000C4430080C4430000C5430080C5430000C6430080C6430000C7430080C7430000C8430080C8430000C9430080C9430000CA430080CA430000CB430080CB430000CC430080CC430000CD430080CD430000CE430080CE430000CF430080CF430000D0430080D0430000D1430080D1430000D2430080D2430000D3430080D3430000D4430080D4430000D5430080D5430000D6430080D6430000D7430080D7430000D8430080D8430000D9430080D9430000DA430080DA430000DB430080DB430000DC430080DC430000DD430080DD430000DE430080DE430000DF430080DF430000E0430080E0430000E1430080E1430000E2430080E2430000E3430080E3430000E4430080E4430000E5430080E5430000E6430080E6430000E7430080E7430000E8430080E8430000E9430080E9430000EA430080EA430000EB430080EB430000EC430080EC430000ED430080ED430000EE430080EE430000EF430080EF430000F0430080F0430000F1430080F1430000F2430080F2430000F3430080F3430000F4430080F4430000F5430080F5430000F6430080F6430000F7430080F7430000F8430080F8430000F9430080F9430000FA430080FA430000FB430080FB430000FC430080FC430000FD430080FD430000FE430080FE430000FF430080FF4300000044004000440080004400C0004400000144004001440080014400C0014400000244004002440080024400C0024400000344004003440080034400C0034400000444004004440080044400C0044400000544004005440080054400C0054400000644004006440080064400C0064400000744004007440080074400C0074400000844004008440080084400C0084400000944004009440080094400C0094400000A4400400A4400800A4400C00A4400000B4400400B4400800B4400C00B4400000C4400400C4400800C4400C00C4400000D4400400D4400800D4400C00D4400000E4400400E4400800E4400C00E4400000F4400400F4400800F4400C00F4400001044004010440080104400C0104400001144004011440080114400C0114400001244004012440080124400C0124400001344004013440080134400C0134400001444004014440080144400C0144400001544004015440080154400C0154400001644004016440080164400C0164400001744004017440080174400C0174400001844004018440080184400C0184400001944004019440080194400C0194400001A4400401A4400801A4400C01A4400001B4400401B4400801B4400C01B4400001C4400401C4400801C4400C01C4400001D4400401D4400801D4400C01D4400001E4400401E4400801E4400C01E4400001F4400401F4400801F4400C01F44"> : tensor<1x1x640xf32>}> : () -> tensor<1x1x640xf32>
    %4 = "ttir.constant"() <{value = dense<3.25520843E-4> : tensor<1x640xf32>}> : () -> tensor<1x640xf32>
    %5 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1x640x3072xf32>}> : () -> tensor<1x640x3072xf32>
    %6 = "ttir.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %7 = "ttir.constant"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F0000000000000080000000000000008100000000000000820000000000000083000000000000008400000000000000850000000000000086000000000000008700000000000000880000000000000089000000000000008A000000000000008B000000000000008C000000000000008D000000000000008E000000000000008F0000000000000090000000000000009100000000000000920000000000000093000000000000009400000000000000950000000000000096000000000000009700000000000000980000000000000099000000000000009A000000000000009B000000000000009C000000000000009D000000000000009E000000000000009F00000000000000A000000000000000A100000000000000A200000000000000A300000000000000A400000000000000A500000000000000A600000000000000A700000000000000A800000000000000A900000000000000AA00000000000000AB00000000000000AC00000000000000AD00000000000000AE00000000000000AF00000000000000B000000000000000B100000000000000B200000000000000B300000000000000B400000000000000B500000000000000B600000000000000B700000000000000B800000000000000B900000000000000BA00000000000000BB00000000000000BC00000000000000BD00000000000000BE00000000000000BF00000000000000C000000000000000C100000000000000C200000000000000C300000000000000C400000000000000C500000000000000C600000000000000C700000000000000C800000000000000C900000000000000CA00000000000000CB00000000000000CC00000000000000CD00000000000000CE00000000000000CF00000000000000D000000000000000D100000000000000D200000000000000D300000000000000D400000000000000D500000000000000D600000000000000D700000000000000D800000000000000D900000000000000DA00000000000000DB00000000000000DC00000000000000DD00000000000000DE00000000000000DF00000000000000E000000000000000E100000000000000E200000000000000E300000000000000E400000000000000E500000000000000E600000000000000E700000000000000E800000000000000E900000000000000EA00000000000000EB00000000000000EC00000000000000ED00000000000000EE00000000000000EF00000000000000F000000000000000F100000000000000F200000000000000F300000000000000F400000000000000F500000000000000F600000000000000F700000000000000F800000000000000F900000000000000FA00000000000000FB00000000000000FC00000000000000FD00000000000000FE00000000000000FF0000000000000000010000000000000101000000000000020100000000000003010000000000000401000000000000050100000000000006010000000000000701000000000000080100000000000009010000000000000A010000000000000B010000000000000C010000000000000D010000000000000E010000000000000F0100000000000010010000000000001101000000000000120100000000000013010000000000001401000000000000150100000000000016010000000000001701000000000000180100000000000019010000000000001A010000000000001B010000000000001C010000000000001D010000000000001E010000000000001F0100000000000020010000000000002101000000000000220100000000000023010000000000002401000000000000250100000000000026010000000000002701000000000000280100000000000029010000000000002A010000000000002B010000000000002C010000000000002D010000000000002E010000000000002F0100000000000030010000000000003101000000000000320100000000000033010000000000003401000000000000350100000000000036010000000000003701000000000000380100000000000039010000000000003A010000000000003B010000000000003C010000000000003D010000000000003E010000000000003F0100000000000040010000000000004101000000000000420100000000000043010000000000004401000000000000450100000000000046010000000000004701000000000000480100000000000049010000000000004A010000000000004B010000000000004C010000000000004D010000000000004E010000000000004F0100000000000050010000000000005101000000000000520100000000000053010000000000005401000000000000550100000000000056010000000000005701000000000000580100000000000059010000000000005A010000000000005B010000000000005C010000000000005D010000000000005E010000000000005F0100000000000060010000000000006101000000000000620100000000000063010000000000006401000000000000650100000000000066010000000000006701000000000000680100000000000069010000000000006A010000000000006B010000000000006C010000000000006D010000000000006E010000000000006F0100000000000070010000000000007101000000000000720100000000000073010000000000007401000000000000750100000000000076010000000000007701000000000000780100000000000079010000000000007A010000000000007B010000000000007C010000000000007D010000000000007E010000000000007F0100000000000080010000000000008101000000000000820100000000000083010000000000008401000000000000850100000000000086010000000000008701000000000000880100000000000089010000000000008A010000000000008B010000000000008C010000000000008D010000000000008E010000000000008F0100000000000090010000000000009101000000000000920100000000000093010000000000009401000000000000950100000000000096010000000000009701000000000000980100000000000099010000000000009A010000000000009B010000000000009C010000000000009D010000000000009E010000000000009F01000000000000A001000000000000A101000000000000A201000000000000A301000000000000A401000000000000A501000000000000A601000000000000A701000000000000A801000000000000A901000000000000AA01000000000000AB01000000000000AC01000000000000AD01000000000000AE01000000000000AF01000000000000B001000000000000B101000000000000B201000000000000B301000000000000B401000000000000B501000000000000B601000000000000B701000000000000B801000000000000B901000000000000BA01000000000000BB01000000000000BC01000000000000BD01000000000000BE01000000000000BF01000000000000C001000000000000C101000000000000C201000000000000C301000000000000C401000000000000C501000000000000C601000000000000C701000000000000C801000000000000C901000000000000CA01000000000000CB01000000000000CC01000000000000CD01000000000000CE01000000000000CF01000000000000D001000000000000D101000000000000D201000000000000D301000000000000D401000000000000D501000000000000D601000000000000D701000000000000D801000000000000D901000000000000DA01000000000000DB01000000000000DC01000000000000DD01000000000000DE01000000000000DF01000000000000E001000000000000E101000000000000E201000000000000E301000000000000E401000000000000E501000000000000E601000000000000E701000000000000E801000000000000E901000000000000EA01000000000000EB01000000000000EC01000000000000ED01000000000000EE01000000000000EF01000000000000F001000000000000F101000000000000F201000000000000F301000000000000F401000000000000F501000000000000F601000000000000F701000000000000F801000000000000F901000000000000FA01000000000000FB01000000000000FC01000000000000FD01000000000000FE01000000000000FF0100000000000000020000000000000102000000000000020200000000000003020000000000000402000000000000050200000000000006020000000000000702000000000000080200000000000009020000000000000A020000000000000B020000000000000C020000000000000D020000000000000E020000000000000F0200000000000010020000000000001102000000000000120200000000000013020000000000001402000000000000150200000000000016020000000000001702000000000000180200000000000019020000000000001A020000000000001B020000000000001C020000000000001D020000000000001E020000000000001F0200000000000020020000000000002102000000000000220200000000000023020000000000002402000000000000250200000000000026020000000000002702000000000000280200000000000029020000000000002A020000000000002B020000000000002C020000000000002D020000000000002E020000000000002F0200000000000030020000000000003102000000000000320200000000000033020000000000003402000000000000350200000000000036020000000000003702000000000000380200000000000039020000000000003A020000000000003B020000000000003C020000000000003D020000000000003E020000000000003F0200000000000040020000000000004102000000000000420200000000000043020000000000004402000000000000450200000000000046020000000000004702000000000000480200000000000049020000000000004A020000000000004B020000000000004C020000000000004D020000000000004E020000000000004F0200000000000050020000000000005102000000000000520200000000000053020000000000005402000000000000550200000000000056020000000000005702000000000000580200000000000059020000000000005A020000000000005B020000000000005C020000000000005D020000000000005E020000000000005F0200000000000060020000000000006102000000000000620200000000000063020000000000006402000000000000650200000000000066020000000000006702000000000000680200000000000069020000000000006A020000000000006B020000000000006C020000000000006D020000000000006E020000000000006F0200000000000070020000000000007102000000000000720200000000000073020000000000007402000000000000750200000000000076020000000000007702000000000000780200000000000079020000000000007A020000000000007B020000000000007C020000000000007D020000000000007E020000000000007F02000000000000"> : tensor<640xi64>}> : () -> tensor<640xi64>
    %8 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %9 = ttir.empty() : tensor<1x1x3072xbf16>
    %10 = "ttir.reshape"(%arg4, %9) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %11 = ttir.empty() : tensor<3072xbf16>
    %12 = "ttir.reshape"(%10, %11) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %13 = ttir.empty() : tensor<3072xf32>
    %14 = "ttir.typecast"(%12, %13) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %15 = ttir.empty() : tensor<1x1x3072xf32>
    %16 = "ttir.reshape"(%14, %15) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %17 = ttir.empty() : tensor<1x640x3072xf32>
    %18 = "ttir.broadcast"(%16, %17) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %19 = ttir.empty() : tensor<1x128256x3072xbf16>
    %20 = "ttir.reshape"(%arg3, %19) <{shape = [1 : i32, 128256 : i32, 3072 : i32]}> : (tensor<128256x3072xbf16>, tensor<1x128256x3072xbf16>) -> tensor<1x128256x3072xbf16>
    %21 = ttir.empty() : tensor<128256x3072xbf16>
    %22 = "ttir.reshape"(%20, %21) <{shape = [128256 : i32, 3072 : i32]}> : (tensor<1x128256x3072xbf16>, tensor<128256x3072xbf16>) -> tensor<128256x3072xbf16>
    %23 = ttir.empty() : tensor<1x1x640xi64>
    %24 = "ttir.reshape"(%arg2, %23) <{shape = [1 : i32, 1 : i32, 640 : i32]}> : (tensor<1x640xi64>, tensor<1x1x640xi64>) -> tensor<1x1x640xi64>
    %25 = ttir.empty() : tensor<640xi64>
    %26 = "ttir.reshape"(%24, %25) <{shape = [640 : i32]}> : (tensor<1x1x640xi64>, tensor<640xi64>) -> tensor<640xi64>
    %27 = ttir.empty() : tensor<640xui32>
    %28 = "ttir.typecast"(%26, %27) <{conservative_folding = false}> : (tensor<640xi64>, tensor<640xui32>) -> tensor<640xui32>
    %29 = ttir.empty() : tensor<640x3072xbf16>
    %30 = "ttir.gather"(%22, %28, %29) <{collapsed_slice_dims = array<i64: 0>, index_vector_dim = 1 : si64, indices_are_sorted = false, offset_dims = array<i64: 1>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 3072>, start_index_map = array<i64: 0>, start_indices_batching_dims = array<i64>}> : (tensor<128256x3072xbf16>, tensor<640xui32>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %31 = ttir.empty() : tensor<1x640x3072xbf16>
    %32 = "ttir.reshape"(%30, %31) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %33 = ttir.empty() : tensor<1x640x3072xf32>
    %34 = "ttir.typecast"(%32, %33) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %35 = ttir.empty() : tensor<1x640x3072xf32>
    %36 = "ttir.pow"(%34, %5, %35) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %37 = ttir.empty() : tensor<1x640xf32>
    %38 = "ttir.sum"(%36, %37) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %39 = ttir.empty() : tensor<1x640xf32>
    %40 = "ttir.multiply"(%38, %4, %39) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %41 = ttir.empty() : tensor<1x640x1xf32>
    %42 = "ttir.reshape"(%40, %41) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %43 = ttir.empty() : tensor<1x1x1xf32>
    %44 = "ttir.reshape"(%arg1, %43) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %45 = ttir.empty() : tensor<1x640x1xf32>
    %46 = "ttir.broadcast"(%44, %45) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %47 = ttir.empty() : tensor<1x640x1xf32>
    %48 = "ttir.add"(%42, %46, %47) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %49 = ttir.empty() : tensor<1x640x1xf32>
    %50 = "ttir.rsqrt"(%48, %49) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %51 = ttir.empty() : tensor<1x640xf32>
    %52 = "ttir.reshape"(%50, %51) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %53 = ttir.empty() : tensor<1x640x1xf32>
    %54 = "ttir.reshape"(%52, %53) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %55 = ttir.empty() : tensor<1x640x3072xf32>
    %56 = "ttir.broadcast"(%54, %55) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %57 = ttir.empty() : tensor<1x640x3072xf32>
    %58 = "ttir.multiply"(%34, %56, %57) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %59 = ttir.empty() : tensor<1x640x3072xbf16>
    %60 = "ttir.typecast"(%58, %59) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %61 = ttir.empty() : tensor<1x640x3072xf32>
    %62 = "ttir.typecast"(%60, %61) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %63 = ttir.empty() : tensor<1x640x3072xf32>
    %64 = "ttir.multiply"(%18, %62, %63) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %65 = ttir.empty() : tensor<1x640x3072xbf16>
    %66 = "ttir.typecast"(%64, %65) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %67 = ttir.empty() : tensor<640x3072xbf16>
    %68 = "ttir.reshape"(%66, %67) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %69 = ttir.empty() : tensor<1x1024x3072xbf16>
    %70 = "ttir.reshape"(%arg0, %69) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %71 = ttir.empty() : tensor<1024x3072xbf16>
    %72 = "ttir.reshape"(%70, %71) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %73 = ttir.empty() : tensor<3072x1024xbf16>
    %74 = "ttir.permute"(%72, %73) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %75 = "ttir.dot_general"(%68, %74) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %76 = ttir.empty() : tensor<1x640x8x128xbf16>
    %77 = "ttir.reshape"(%75, %76) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %78 = ttir.empty() : tensor<1x8x640x128xbf16>
    %79 = "ttir.permute"(%77, %78) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %80 = ttir.empty() : tensor<1x1x3072xbf16>
    %81 = "ttir.reshape"(%arg17, %80) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %82 = ttir.empty() : tensor<3072xbf16>
    %83 = "ttir.reshape"(%81, %82) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %84 = ttir.empty() : tensor<3072xf32>
    %85 = "ttir.typecast"(%83, %84) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %86 = ttir.empty() : tensor<1x1x3072xf32>
    %87 = "ttir.reshape"(%85, %86) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %88 = ttir.empty() : tensor<1x640x3072xf32>
    %89 = "ttir.broadcast"(%87, %88) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %90 = ttir.empty() : tensor<1x3072x3072xbf16>
    %91 = "ttir.reshape"(%arg14, %90) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %92 = ttir.empty() : tensor<3072x3072xbf16>
    %93 = "ttir.reshape"(%91, %92) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %94 = ttir.empty() : tensor<3072x3072xbf16>
    %95 = "ttir.permute"(%93, %94) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %96 = "ttir.dot_general"(%68, %95) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %97 = ttir.empty() : tensor<1x640x24x128xbf16>
    %98 = "ttir.reshape"(%96, %97) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %99 = ttir.empty() : tensor<1x24x640x128xbf16>
    %100 = "ttir.permute"(%98, %99) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %101 = ttir.empty() : tensor<1x24x640x128xf32>
    %102 = "ttir.typecast"(%100, %101) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %103 = ttir.empty() : tensor<1x1x64xf32>
    %104 = "ttir.reshape"(%arg12, %103) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<64xf32>, tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
    %105 = ttir.empty() : tensor<1x64x1xf32>
    %106 = "ttir.reshape"(%104, %105) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<1x1x64xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %107 = "ttir.dot_general"(%106, %3) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<1x64x1xf32>, tensor<1x1x640xf32>) -> tensor<1x64x640xf32>
    %108 = ttir.empty() : tensor<1x640x64xf32>
    %109 = "ttir.permute"(%107, %108) <{permutation = array<i64: 0, 2, 1>}> : (tensor<1x64x640xf32>, tensor<1x640x64xf32>) -> tensor<1x640x64xf32>
    %110 = ttir.empty() : tensor<1x640x128xf32>
    %111 = "ttir.concat"(%109, %109, %110) <{dim = 2 : si32}> : (tensor<1x640x64xf32>, tensor<1x640x64xf32>, tensor<1x640x128xf32>) -> tensor<1x640x128xf32>
    %112 = ttir.empty() : tensor<1x640x128xf32>
    %113 = "ttir.cos"(%111, %112) : (tensor<1x640x128xf32>, tensor<1x640x128xf32>) -> tensor<1x640x128xf32>
    %114 = ttir.empty() : tensor<1x640x128xbf16>
    %115 = "ttir.typecast"(%113, %114) <{conservative_folding = false}> : (tensor<1x640x128xf32>, tensor<1x640x128xbf16>) -> tensor<1x640x128xbf16>
    %116 = ttir.empty() : tensor<1x1x640x128xbf16>
    %117 = "ttir.reshape"(%115, %116) <{shape = [1 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x640x128xbf16>, tensor<1x1x640x128xbf16>) -> tensor<1x1x640x128xbf16>
    %118 = ttir.empty() : tensor<1x1x640x128xf32>
    %119 = "ttir.typecast"(%117, %118) <{conservative_folding = false}> : (tensor<1x1x640x128xbf16>, tensor<1x1x640x128xf32>) -> tensor<1x1x640x128xf32>
    %120 = ttir.empty() : tensor<1x640x128xf32>
    %121 = "ttir.reshape"(%119, %120) <{shape = [1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x1x640x128xf32>, tensor<1x640x128xf32>) -> tensor<1x640x128xf32>
    %122 = ttir.empty() : tensor<1x1x640x128xf32>
    %123 = "ttir.reshape"(%121, %122) <{shape = [1 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x640x128xf32>, tensor<1x1x640x128xf32>) -> tensor<1x1x640x128xf32>
    %124 = ttir.empty() : tensor<1x24x640x128xf32>
    %125 = "ttir.broadcast"(%123, %124) <{broadcast_dimensions = array<i64: 1, 24, 1, 1>}> : (tensor<1x1x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %126 = ttir.empty() : tensor<1x24x640x128xf32>
    %127 = "ttir.multiply"(%102, %125, %126) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %128 = ttir.empty() : tensor<1x24x640x128xbf16>
    %129 = "ttir.typecast"(%127, %128) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %130 = ttir.empty() : tensor<1x24x640x64xbf16>
    %131 = "ttir.slice_static"(%100, %130) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %132 = ttir.empty() : tensor<1x24x640x64xbf16>
    %133 = "ttir.neg"(%131, %132) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %134 = ttir.empty() : tensor<1x24x640x64xbf16>
    %135 = "ttir.slice_static"(%100, %134) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %136 = ttir.empty() : tensor<1x24x640x128xbf16>
    %137 = "ttir.concat"(%133, %135, %136) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %138 = ttir.empty() : tensor<1x24x640x128xf32>
    %139 = "ttir.typecast"(%137, %138) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %140 = ttir.empty() : tensor<1x640x128xf32>
    %141 = "ttir.sin"(%111, %140) : (tensor<1x640x128xf32>, tensor<1x640x128xf32>) -> tensor<1x640x128xf32>
    %142 = ttir.empty() : tensor<1x640x128xbf16>
    %143 = "ttir.typecast"(%141, %142) <{conservative_folding = false}> : (tensor<1x640x128xf32>, tensor<1x640x128xbf16>) -> tensor<1x640x128xbf16>
    %144 = ttir.empty() : tensor<1x1x640x128xbf16>
    %145 = "ttir.reshape"(%143, %144) <{shape = [1 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x640x128xbf16>, tensor<1x1x640x128xbf16>) -> tensor<1x1x640x128xbf16>
    %146 = ttir.empty() : tensor<1x1x640x128xf32>
    %147 = "ttir.typecast"(%145, %146) <{conservative_folding = false}> : (tensor<1x1x640x128xbf16>, tensor<1x1x640x128xf32>) -> tensor<1x1x640x128xf32>
    %148 = ttir.empty() : tensor<1x640x128xf32>
    %149 = "ttir.reshape"(%147, %148) <{shape = [1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x1x640x128xf32>, tensor<1x640x128xf32>) -> tensor<1x640x128xf32>
    %150 = ttir.empty() : tensor<1x1x640x128xf32>
    %151 = "ttir.reshape"(%149, %150) <{shape = [1 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x640x128xf32>, tensor<1x1x640x128xf32>) -> tensor<1x1x640x128xf32>
    %152 = ttir.empty() : tensor<1x24x640x128xf32>
    %153 = "ttir.broadcast"(%151, %152) <{broadcast_dimensions = array<i64: 1, 24, 1, 1>}> : (tensor<1x1x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %154 = ttir.empty() : tensor<1x24x640x128xf32>
    %155 = "ttir.multiply"(%139, %153, %154) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %156 = ttir.empty() : tensor<1x24x640x128xbf16>
    %157 = "ttir.typecast"(%155, %156) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %158 = ttir.empty() : tensor<1x24x640x128xbf16>
    %159 = "ttir.add"(%129, %157, %158) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %160 = ttir.empty() : tensor<24x640x128xbf16>
    %161 = "ttir.reshape"(%159, %160) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %162 = ttir.empty() : tensor<1x1024x3072xbf16>
    %163 = "ttir.reshape"(%arg13, %162) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %164 = ttir.empty() : tensor<1024x3072xbf16>
    %165 = "ttir.reshape"(%163, %164) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %166 = ttir.empty() : tensor<3072x1024xbf16>
    %167 = "ttir.permute"(%165, %166) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %168 = "ttir.dot_general"(%68, %167) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %169 = ttir.empty() : tensor<1x640x8x128xbf16>
    %170 = "ttir.reshape"(%168, %169) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %171 = ttir.empty() : tensor<1x8x640x128xbf16>
    %172 = "ttir.permute"(%170, %171) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %173 = ttir.empty() : tensor<1x8x640x128xf32>
    %174 = "ttir.typecast"(%172, %173) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %175 = ttir.empty() : tensor<1x1x640x128xf32>
    %176 = "ttir.reshape"(%121, %175) <{shape = [1 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x640x128xf32>, tensor<1x1x640x128xf32>) -> tensor<1x1x640x128xf32>
    %177 = ttir.empty() : tensor<1x8x640x128xf32>
    %178 = "ttir.broadcast"(%176, %177) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %179 = ttir.empty() : tensor<1x8x640x128xf32>
    %180 = "ttir.multiply"(%174, %178, %179) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %181 = ttir.empty() : tensor<1x8x640x128xbf16>
    %182 = "ttir.typecast"(%180, %181) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %183 = ttir.empty() : tensor<1x8x640x64xbf16>
    %184 = "ttir.slice_static"(%172, %183) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %185 = ttir.empty() : tensor<1x8x640x64xbf16>
    %186 = "ttir.neg"(%184, %185) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %187 = ttir.empty() : tensor<1x8x640x64xbf16>
    %188 = "ttir.slice_static"(%172, %187) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %189 = ttir.empty() : tensor<1x8x640x128xbf16>
    %190 = "ttir.concat"(%186, %188, %189) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %191 = ttir.empty() : tensor<1x8x640x128xf32>
    %192 = "ttir.typecast"(%190, %191) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %193 = ttir.empty() : tensor<1x1x640x128xf32>
    %194 = "ttir.reshape"(%149, %193) <{shape = [1 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x640x128xf32>, tensor<1x1x640x128xf32>) -> tensor<1x1x640x128xf32>
    %195 = ttir.empty() : tensor<1x8x640x128xf32>
    %196 = "ttir.broadcast"(%194, %195) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %197 = ttir.empty() : tensor<1x8x640x128xf32>
    %198 = "ttir.multiply"(%192, %196, %197) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %199 = ttir.empty() : tensor<1x8x640x128xbf16>
    %200 = "ttir.typecast"(%198, %199) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %201 = ttir.empty() : tensor<1x8x640x128xbf16>
    %202 = "ttir.add"(%182, %200, %201) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %203 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %204 = "ttir.reshape"(%202, %203) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %205 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %206 = "ttir.broadcast"(%204, %205) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %207 = ttir.empty() : tensor<1x24x640x128xbf16>
    %208 = "ttir.reshape"(%206, %207) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %209 = ttir.empty() : tensor<1x24x128x640xbf16>
    %210 = "ttir.permute"(%208, %209) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %211 = ttir.empty() : tensor<24x128x640xbf16>
    %212 = "ttir.reshape"(%210, %211) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %213 = "ttir.dot_general"(%161, %212) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %214 = ttir.empty() : tensor<1x24x640x640xbf16>
    %215 = "ttir.reshape"(%213, %214) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %216 = ttir.empty() : tensor<1x24x640x640xf32>
    %217 = "ttir.typecast"(%215, %216) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %218 = ttir.empty() : tensor<1x1x1x1xf32>
    %219 = "ttir.reshape"(%arg11, %218) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %220 = ttir.empty() : tensor<1x24x640x640xf32>
    %221 = "ttir.broadcast"(%219, %220) <{broadcast_dimensions = array<i64: 1, 24, 640, 640>}> : (tensor<1x1x1x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %222 = ttir.empty() : tensor<1x24x640x640xf32>
    %223 = "ttir.multiply"(%217, %221, %222) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %224 = ttir.empty() : tensor<1x24x640x640xbf16>
    %225 = "ttir.typecast"(%223, %224) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %226 = ttir.empty() : tensor<1x640xi64>
    %227 = "ttir.reshape"(%7, %226) <{shape = [1 : i32, 640 : i32]}> : (tensor<640xi64>, tensor<1x640xi64>) -> tensor<1x640xi64>
    %228 = ttir.empty() : tensor<640x640xi64>
    %229 = "ttir.broadcast"(%227, %228) <{broadcast_dimensions = array<i64: 640, 1>}> : (tensor<1x640xi64>, tensor<640x640xi64>) -> tensor<640x640xi64>
    %230 = ttir.empty() : tensor<640x1xi64>
    %231 = "ttir.reshape"(%7, %230) <{shape = [640 : i32, 1 : i32]}> : (tensor<640xi64>, tensor<640x1xi64>) -> tensor<640x1xi64>
    %232 = ttir.empty() : tensor<640x640xi64>
    %233 = "ttir.broadcast"(%231, %232) <{broadcast_dimensions = array<i64: 1, 640>}> : (tensor<640x1xi64>, tensor<640x640xi64>) -> tensor<640x640xi64>
    %234 = ttir.empty() : tensor<640x640xi64>
    %235 = "ttir.subtract"(%229, %233, %234) : (tensor<640x640xi64>, tensor<640x640xi64>, tensor<640x640xi64>) -> tensor<640x640xi64>
    %236 = ttir.empty() : tensor<640x640xi1>
    %237 = "ttir.ge"(%235, %2, %236) : (tensor<640x640xi64>, tensor<640x640xi64>, tensor<640x640xi1>) -> tensor<640x640xi1>
    %238 = ttir.empty() : tensor<1x1xbf16>
    %239 = "ttir.reshape"(%arg9, %238) <{shape = [1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %240 = ttir.empty() : tensor<640x640xbf16>
    %241 = "ttir.broadcast"(%239, %240) <{broadcast_dimensions = array<i64: 640, 640>}> : (tensor<1x1xbf16>, tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %242 = ttir.empty() : tensor<640x640xbf16>
    %243 = "ttir.where"(%237, %241, %1, %242) : (tensor<640x640xi1>, tensor<640x640xbf16>, tensor<640x640xbf16>, tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %244 = ttir.empty() : tensor<640x640xf32>
    %245 = "ttir.typecast"(%243, %244) <{conservative_folding = false}> : (tensor<640x640xbf16>, tensor<640x640xf32>) -> tensor<640x640xf32>
    %246 = ttir.empty() : tensor<640x640xi1>
    %247 = "ttir.gt"(%229, %233, %246) : (tensor<640x640xi64>, tensor<640x640xi64>, tensor<640x640xi1>) -> tensor<640x640xi1>
    %248 = ttir.empty() : tensor<640x640xf32>
    %249 = "ttir.typecast"(%247, %248) <{conservative_folding = false}> : (tensor<640x640xi1>, tensor<640x640xf32>) -> tensor<640x640xf32>
    %250 = ttir.empty() : tensor<640x640xf32>
    %251 = "ttir.multiply"(%245, %249, %250) : (tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>) -> tensor<640x640xf32>
    %252 = ttir.empty() : tensor<640x640xbf16>
    %253 = "ttir.typecast"(%251, %252) <{conservative_folding = false}> : (tensor<640x640xf32>, tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %254 = ttir.empty() : tensor<1x1x640x640xbf16>
    %255 = "ttir.reshape"(%253, %254) <{shape = [1 : i32, 1 : i32, 640 : i32, 640 : i32]}> : (tensor<640x640xbf16>, tensor<1x1x640x640xbf16>) -> tensor<1x1x640x640xbf16>
    %256 = ttir.empty() : tensor<1x1x640xi64>
    %257 = "ttir.reshape"(%arg10, %256) <{shape = [1 : i32, 1 : i32, 640 : i32]}> : (tensor<1x640xi64>, tensor<1x1x640xi64>) -> tensor<1x1x640xi64>
    %258 = ttir.empty() : tensor<1x1x1x640xi64>
    %259 = "ttir.reshape"(%257, %258) <{shape = [1 : i32, 1 : i32, 1 : i32, 640 : i32]}> : (tensor<1x1x640xi64>, tensor<1x1x1x640xi64>) -> tensor<1x1x1x640xi64>
    %260 = ttir.empty() : tensor<1x1x1x640xbf16>
    %261 = "ttir.typecast"(%259, %260) <{conservative_folding = false}> : (tensor<1x1x1x640xi64>, tensor<1x1x1x640xbf16>) -> tensor<1x1x1x640xbf16>
    %262 = ttir.empty() : tensor<1x1x640xbf16>
    %263 = "ttir.reshape"(%261, %262) <{shape = [1 : i32, 1 : i32, 640 : i32]}> : (tensor<1x1x1x640xbf16>, tensor<1x1x640xbf16>) -> tensor<1x1x640xbf16>
    %264 = ttir.empty() : tensor<1x1x1x640xbf16>
    %265 = "ttir.reshape"(%263, %264) <{shape = [1 : i32, 1 : i32, 1 : i32, 640 : i32]}> : (tensor<1x1x640xbf16>, tensor<1x1x1x640xbf16>) -> tensor<1x1x1x640xbf16>
    %266 = ttir.empty() : tensor<1x1x640x640xbf16>
    %267 = "ttir.broadcast"(%265, %266) <{broadcast_dimensions = array<i64: 1, 1, 640, 1>}> : (tensor<1x1x1x640xbf16>, tensor<1x1x640x640xbf16>) -> tensor<1x1x640x640xbf16>
    %268 = ttir.empty() : tensor<1x1x640x640xbf16>
    %269 = "ttir.add"(%255, %267, %268) : (tensor<1x1x640x640xbf16>, tensor<1x1x640x640xbf16>, tensor<1x1x640x640xbf16>) -> tensor<1x1x640x640xbf16>
    %270 = ttir.empty() : tensor<1x1x640x640xi1>
    %271 = "ttir.eq"(%269, %0, %270) : (tensor<1x1x640x640xbf16>, tensor<1x1x640x640xbf16>, tensor<1x1x640x640xi1>) -> tensor<1x1x640x640xi1>
    %272 = ttir.empty() : tensor<1x1xbf16>
    %273 = "ttir.reshape"(%arg9, %272) <{shape = [1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %274 = ttir.empty() : tensor<1x1x1x1xbf16>
    %275 = "ttir.reshape"(%273, %274) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x1x1x1xbf16>
    %276 = ttir.empty() : tensor<1x1x640x640xbf16>
    %277 = "ttir.broadcast"(%275, %276) <{broadcast_dimensions = array<i64: 1, 1, 640, 640>}> : (tensor<1x1x1x1xbf16>, tensor<1x1x640x640xbf16>) -> tensor<1x1x640x640xbf16>
    %278 = ttir.empty() : tensor<1x1x640x640xbf16>
    %279 = "ttir.where"(%271, %277, %255, %278) : (tensor<1x1x640x640xi1>, tensor<1x1x640x640xbf16>, tensor<1x1x640x640xbf16>, tensor<1x1x640x640xbf16>) -> tensor<1x1x640x640xbf16>
    %280 = ttir.empty() : tensor<1x640x640xbf16>
    %281 = "ttir.reshape"(%279, %280) <{shape = [1 : i32, 640 : i32, 640 : i32]}> : (tensor<1x1x640x640xbf16>, tensor<1x640x640xbf16>) -> tensor<1x640x640xbf16>
    %282 = ttir.empty() : tensor<1x1x640x640xbf16>
    %283 = "ttir.reshape"(%281, %282) <{shape = [1 : i32, 1 : i32, 640 : i32, 640 : i32]}> : (tensor<1x640x640xbf16>, tensor<1x1x640x640xbf16>) -> tensor<1x1x640x640xbf16>
    %284 = ttir.empty() : tensor<1x24x640x640xbf16>
    %285 = "ttir.broadcast"(%283, %284) <{broadcast_dimensions = array<i64: 1, 24, 1, 1>}> : (tensor<1x1x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %286 = ttir.empty() : tensor<1x24x640x640xbf16>
    %287 = "ttir.add"(%225, %285, %286) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %288 = ttir.empty() : tensor<1x24x640x640xf32>
    %289 = "ttir.typecast"(%287, %288) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %290 = ttir.empty() : tensor<1x24x640xf32>
    %291 = "ttir.max"(%289, %290) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %292 = ttir.empty() : tensor<1x24x640x1xf32>
    %293 = "ttir.reshape"(%291, %292) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %294 = ttir.empty() : tensor<1x24x640x640xf32>
    %295 = "ttir.broadcast"(%293, %294) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %296 = ttir.empty() : tensor<1x24x640x640xf32>
    %297 = "ttir.subtract"(%289, %295, %296) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %298 = ttir.empty() : tensor<1x24x640x640xf32>
    %299 = "ttir.exp"(%297, %298) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %300 = ttir.empty() : tensor<1x24x640xf32>
    %301 = "ttir.sum"(%299, %300) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %302 = ttir.empty() : tensor<1x24x640x1xf32>
    %303 = "ttir.reshape"(%301, %302) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %304 = ttir.empty() : tensor<1x24x640x640xf32>
    %305 = "ttir.broadcast"(%303, %304) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %306 = ttir.empty() : tensor<1x24x640x640xf32>
    %307 = "ttir.div"(%299, %305, %306) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %308 = ttir.empty() : tensor<1x24x640x640xbf16>
    %309 = "ttir.typecast"(%307, %308) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %310 = ttir.empty() : tensor<24x640x640xbf16>
    %311 = "ttir.reshape"(%309, %310) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %312 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %313 = "ttir.reshape"(%79, %312) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %314 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %315 = "ttir.broadcast"(%313, %314) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %316 = ttir.empty() : tensor<24x640x128xbf16>
    %317 = "ttir.reshape"(%315, %316) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %318 = "ttir.dot_general"(%311, %317) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %319 = ttir.empty() : tensor<1x24x640x128xbf16>
    %320 = "ttir.reshape"(%318, %319) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %321 = ttir.empty() : tensor<1x640x24x128xbf16>
    %322 = "ttir.permute"(%320, %321) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %323 = ttir.empty() : tensor<640x3072xbf16>
    %324 = "ttir.reshape"(%322, %323) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %325 = ttir.empty() : tensor<1x3072x3072xbf16>
    %326 = "ttir.reshape"(%arg8, %325) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %327 = ttir.empty() : tensor<3072x3072xbf16>
    %328 = "ttir.reshape"(%326, %327) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %329 = ttir.empty() : tensor<3072x3072xbf16>
    %330 = "ttir.permute"(%328, %329) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %331 = "ttir.dot_general"(%324, %330) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %332 = ttir.empty() : tensor<1x640x3072xbf16>
    %333 = "ttir.reshape"(%331, %332) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %334 = ttir.empty() : tensor<1x640x3072xbf16>
    %335 = "ttir.add"(%32, %333, %334) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %336 = ttir.empty() : tensor<1x1x3072xbf16>
    %337 = "ttir.reshape"(%arg15, %336) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %338 = ttir.empty() : tensor<3072xbf16>
    %339 = "ttir.reshape"(%337, %338) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %340 = ttir.empty() : tensor<3072xf32>
    %341 = "ttir.typecast"(%339, %340) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %342 = ttir.empty() : tensor<1x1x3072xf32>
    %343 = "ttir.reshape"(%341, %342) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %344 = ttir.empty() : tensor<1x640x3072xf32>
    %345 = "ttir.broadcast"(%343, %344) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %346 = ttir.empty() : tensor<1x640x3072xf32>
    %347 = "ttir.typecast"(%335, %346) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %348 = ttir.empty() : tensor<1x640x3072xf32>
    %349 = "ttir.pow"(%347, %5, %348) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %350 = ttir.empty() : tensor<1x640xf32>
    %351 = "ttir.sum"(%349, %350) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %352 = ttir.empty() : tensor<1x640xf32>
    %353 = "ttir.multiply"(%351, %4, %352) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %354 = ttir.empty() : tensor<1x640x1xf32>
    %355 = "ttir.reshape"(%353, %354) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %356 = ttir.empty() : tensor<1x640x1xf32>
    %357 = "ttir.add"(%355, %46, %356) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %358 = ttir.empty() : tensor<1x640x1xf32>
    %359 = "ttir.rsqrt"(%357, %358) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %360 = ttir.empty() : tensor<1x640xf32>
    %361 = "ttir.reshape"(%359, %360) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %362 = ttir.empty() : tensor<1x640x1xf32>
    %363 = "ttir.reshape"(%361, %362) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %364 = ttir.empty() : tensor<1x640x3072xf32>
    %365 = "ttir.broadcast"(%363, %364) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %366 = ttir.empty() : tensor<1x640x3072xf32>
    %367 = "ttir.multiply"(%347, %365, %366) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %368 = ttir.empty() : tensor<1x640x3072xbf16>
    %369 = "ttir.typecast"(%367, %368) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %370 = ttir.empty() : tensor<1x640x3072xf32>
    %371 = "ttir.typecast"(%369, %370) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %372 = ttir.empty() : tensor<1x640x3072xf32>
    %373 = "ttir.multiply"(%345, %371, %372) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %374 = ttir.empty() : tensor<1x640x3072xbf16>
    %375 = "ttir.typecast"(%373, %374) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %376 = ttir.empty() : tensor<640x3072xbf16>
    %377 = "ttir.reshape"(%375, %376) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %378 = ttir.empty() : tensor<1x8192x3072xbf16>
    %379 = "ttir.reshape"(%arg16, %378) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %380 = ttir.empty() : tensor<8192x3072xbf16>
    %381 = "ttir.reshape"(%379, %380) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %382 = ttir.empty() : tensor<3072x8192xbf16>
    %383 = "ttir.permute"(%381, %382) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %384 = "ttir.dot_general"(%377, %383) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %385 = ttir.empty() : tensor<1x640x8192xbf16>
    %386 = "ttir.reshape"(%384, %385) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %387 = ttir.empty() : tensor<1x640x8192xf32>
    %388 = "ttir.typecast"(%386, %387) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %389 = ttir.empty() : tensor<1x640x8192xbf16>
    %390 = "ttir.sigmoid"(%386, %389) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %391 = ttir.empty() : tensor<1x640x8192xf32>
    %392 = "ttir.typecast"(%390, %391) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %393 = ttir.empty() : tensor<1x640x8192xf32>
    %394 = "ttir.multiply"(%388, %392, %393) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %395 = ttir.empty() : tensor<1x640x8192xbf16>
    %396 = "ttir.typecast"(%394, %395) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %397 = ttir.empty() : tensor<1x640x8192xf32>
    %398 = "ttir.typecast"(%396, %397) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %399 = ttir.empty() : tensor<1x8192x3072xbf16>
    %400 = "ttir.reshape"(%arg7, %399) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %401 = ttir.empty() : tensor<8192x3072xbf16>
    %402 = "ttir.reshape"(%400, %401) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %403 = ttir.empty() : tensor<3072x8192xbf16>
    %404 = "ttir.permute"(%402, %403) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %405 = "ttir.dot_general"(%377, %404) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %406 = ttir.empty() : tensor<1x640x8192xbf16>
    %407 = "ttir.reshape"(%405, %406) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %408 = ttir.empty() : tensor<1x640x8192xf32>
    %409 = "ttir.typecast"(%407, %408) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %410 = ttir.empty() : tensor<1x640x8192xf32>
    %411 = "ttir.multiply"(%398, %409, %410) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %412 = ttir.empty() : tensor<1x640x8192xbf16>
    %413 = "ttir.typecast"(%411, %412) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %414 = ttir.empty() : tensor<640x8192xbf16>
    %415 = "ttir.reshape"(%413, %414) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %416 = ttir.empty() : tensor<1x3072x8192xbf16>
    %417 = "ttir.reshape"(%arg6, %416) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %418 = ttir.empty() : tensor<3072x8192xbf16>
    %419 = "ttir.reshape"(%417, %418) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %420 = ttir.empty() : tensor<8192x3072xbf16>
    %421 = "ttir.permute"(%419, %420) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %422 = "ttir.dot_general"(%415, %421) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %423 = ttir.empty() : tensor<1x640x3072xbf16>
    %424 = "ttir.reshape"(%422, %423) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %425 = ttir.empty() : tensor<1x640x3072xbf16>
    %426 = "ttir.add"(%335, %424, %425) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %427 = ttir.empty() : tensor<1x640x3072xf32>
    %428 = "ttir.typecast"(%426, %427) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %429 = ttir.empty() : tensor<1x640x3072xf32>
    %430 = "ttir.pow"(%428, %5, %429) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %431 = ttir.empty() : tensor<1x640xf32>
    %432 = "ttir.sum"(%430, %431) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %433 = ttir.empty() : tensor<1x640xf32>
    %434 = "ttir.multiply"(%432, %4, %433) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %435 = ttir.empty() : tensor<1x640x1xf32>
    %436 = "ttir.reshape"(%434, %435) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %437 = ttir.empty() : tensor<1x640x1xf32>
    %438 = "ttir.add"(%436, %46, %437) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %439 = ttir.empty() : tensor<1x640x1xf32>
    %440 = "ttir.rsqrt"(%438, %439) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %441 = ttir.empty() : tensor<1x640xf32>
    %442 = "ttir.reshape"(%440, %441) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %443 = ttir.empty() : tensor<1x640x1xf32>
    %444 = "ttir.reshape"(%442, %443) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %445 = ttir.empty() : tensor<1x640x3072xf32>
    %446 = "ttir.broadcast"(%444, %445) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %447 = ttir.empty() : tensor<1x640x3072xf32>
    %448 = "ttir.multiply"(%428, %446, %447) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %449 = ttir.empty() : tensor<1x640x3072xbf16>
    %450 = "ttir.typecast"(%448, %449) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %451 = ttir.empty() : tensor<1x640x3072xf32>
    %452 = "ttir.typecast"(%450, %451) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %453 = ttir.empty() : tensor<1x640x3072xf32>
    %454 = "ttir.multiply"(%89, %452, %453) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %455 = ttir.empty() : tensor<1x640x3072xbf16>
    %456 = "ttir.typecast"(%454, %455) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %457 = ttir.empty() : tensor<640x3072xbf16>
    %458 = "ttir.reshape"(%456, %457) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %459 = ttir.empty() : tensor<1x1024x3072xbf16>
    %460 = "ttir.reshape"(%arg5, %459) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %461 = ttir.empty() : tensor<1024x3072xbf16>
    %462 = "ttir.reshape"(%460, %461) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %463 = ttir.empty() : tensor<3072x1024xbf16>
    %464 = "ttir.permute"(%462, %463) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %465 = "ttir.dot_general"(%458, %464) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %466 = ttir.empty() : tensor<1x640x8x128xbf16>
    %467 = "ttir.reshape"(%465, %466) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %468 = ttir.empty() : tensor<1x8x640x128xbf16>
    %469 = "ttir.permute"(%467, %468) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %470 = ttir.empty() : tensor<1x1x3072xbf16>
    %471 = "ttir.reshape"(%arg26, %470) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %472 = ttir.empty() : tensor<3072xbf16>
    %473 = "ttir.reshape"(%471, %472) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %474 = ttir.empty() : tensor<3072xf32>
    %475 = "ttir.typecast"(%473, %474) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %476 = ttir.empty() : tensor<1x1x3072xf32>
    %477 = "ttir.reshape"(%475, %476) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %478 = ttir.empty() : tensor<1x640x3072xf32>
    %479 = "ttir.broadcast"(%477, %478) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %480 = ttir.empty() : tensor<1x3072x3072xbf16>
    %481 = "ttir.reshape"(%arg23, %480) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %482 = ttir.empty() : tensor<3072x3072xbf16>
    %483 = "ttir.reshape"(%481, %482) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %484 = ttir.empty() : tensor<3072x3072xbf16>
    %485 = "ttir.permute"(%483, %484) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %486 = "ttir.dot_general"(%458, %485) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %487 = ttir.empty() : tensor<1x640x24x128xbf16>
    %488 = "ttir.reshape"(%486, %487) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %489 = ttir.empty() : tensor<1x24x640x128xbf16>
    %490 = "ttir.permute"(%488, %489) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %491 = ttir.empty() : tensor<1x24x640x128xf32>
    %492 = "ttir.typecast"(%490, %491) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %493 = ttir.empty() : tensor<1x24x640x128xf32>
    %494 = "ttir.multiply"(%492, %125, %493) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %495 = ttir.empty() : tensor<1x24x640x128xbf16>
    %496 = "ttir.typecast"(%494, %495) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %497 = ttir.empty() : tensor<1x24x640x64xbf16>
    %498 = "ttir.slice_static"(%490, %497) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %499 = ttir.empty() : tensor<1x24x640x64xbf16>
    %500 = "ttir.neg"(%498, %499) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %501 = ttir.empty() : tensor<1x24x640x64xbf16>
    %502 = "ttir.slice_static"(%490, %501) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %503 = ttir.empty() : tensor<1x24x640x128xbf16>
    %504 = "ttir.concat"(%500, %502, %503) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %505 = ttir.empty() : tensor<1x24x640x128xf32>
    %506 = "ttir.typecast"(%504, %505) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %507 = ttir.empty() : tensor<1x24x640x128xf32>
    %508 = "ttir.multiply"(%506, %153, %507) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %509 = ttir.empty() : tensor<1x24x640x128xbf16>
    %510 = "ttir.typecast"(%508, %509) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %511 = ttir.empty() : tensor<1x24x640x128xbf16>
    %512 = "ttir.add"(%496, %510, %511) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %513 = ttir.empty() : tensor<24x640x128xbf16>
    %514 = "ttir.reshape"(%512, %513) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %515 = ttir.empty() : tensor<1x1024x3072xbf16>
    %516 = "ttir.reshape"(%arg22, %515) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %517 = ttir.empty() : tensor<1024x3072xbf16>
    %518 = "ttir.reshape"(%516, %517) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %519 = ttir.empty() : tensor<3072x1024xbf16>
    %520 = "ttir.permute"(%518, %519) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %521 = "ttir.dot_general"(%458, %520) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %522 = ttir.empty() : tensor<1x640x8x128xbf16>
    %523 = "ttir.reshape"(%521, %522) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %524 = ttir.empty() : tensor<1x8x640x128xbf16>
    %525 = "ttir.permute"(%523, %524) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %526 = ttir.empty() : tensor<1x8x640x128xf32>
    %527 = "ttir.typecast"(%525, %526) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %528 = ttir.empty() : tensor<1x8x640x128xf32>
    %529 = "ttir.multiply"(%527, %178, %528) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %530 = ttir.empty() : tensor<1x8x640x128xbf16>
    %531 = "ttir.typecast"(%529, %530) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %532 = ttir.empty() : tensor<1x8x640x64xbf16>
    %533 = "ttir.slice_static"(%525, %532) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %534 = ttir.empty() : tensor<1x8x640x64xbf16>
    %535 = "ttir.neg"(%533, %534) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %536 = ttir.empty() : tensor<1x8x640x64xbf16>
    %537 = "ttir.slice_static"(%525, %536) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %538 = ttir.empty() : tensor<1x8x640x128xbf16>
    %539 = "ttir.concat"(%535, %537, %538) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %540 = ttir.empty() : tensor<1x8x640x128xf32>
    %541 = "ttir.typecast"(%539, %540) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %542 = ttir.empty() : tensor<1x8x640x128xf32>
    %543 = "ttir.multiply"(%541, %196, %542) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %544 = ttir.empty() : tensor<1x8x640x128xbf16>
    %545 = "ttir.typecast"(%543, %544) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %546 = ttir.empty() : tensor<1x8x640x128xbf16>
    %547 = "ttir.add"(%531, %545, %546) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %548 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %549 = "ttir.reshape"(%547, %548) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %550 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %551 = "ttir.broadcast"(%549, %550) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %552 = ttir.empty() : tensor<1x24x640x128xbf16>
    %553 = "ttir.reshape"(%551, %552) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %554 = ttir.empty() : tensor<1x24x128x640xbf16>
    %555 = "ttir.permute"(%553, %554) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %556 = ttir.empty() : tensor<24x128x640xbf16>
    %557 = "ttir.reshape"(%555, %556) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %558 = "ttir.dot_general"(%514, %557) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %559 = ttir.empty() : tensor<1x24x640x640xbf16>
    %560 = "ttir.reshape"(%558, %559) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %561 = ttir.empty() : tensor<1x24x640x640xf32>
    %562 = "ttir.typecast"(%560, %561) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %563 = ttir.empty() : tensor<1x24x640x640xf32>
    %564 = "ttir.multiply"(%562, %221, %563) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %565 = ttir.empty() : tensor<1x24x640x640xbf16>
    %566 = "ttir.typecast"(%564, %565) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %567 = ttir.empty() : tensor<1x24x640x640xbf16>
    %568 = "ttir.add"(%566, %285, %567) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %569 = ttir.empty() : tensor<1x24x640x640xf32>
    %570 = "ttir.typecast"(%568, %569) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %571 = ttir.empty() : tensor<1x24x640xf32>
    %572 = "ttir.max"(%570, %571) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %573 = ttir.empty() : tensor<1x24x640x1xf32>
    %574 = "ttir.reshape"(%572, %573) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %575 = ttir.empty() : tensor<1x24x640x640xf32>
    %576 = "ttir.broadcast"(%574, %575) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %577 = ttir.empty() : tensor<1x24x640x640xf32>
    %578 = "ttir.subtract"(%570, %576, %577) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %579 = ttir.empty() : tensor<1x24x640x640xf32>
    %580 = "ttir.exp"(%578, %579) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %581 = ttir.empty() : tensor<1x24x640xf32>
    %582 = "ttir.sum"(%580, %581) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %583 = ttir.empty() : tensor<1x24x640x1xf32>
    %584 = "ttir.reshape"(%582, %583) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %585 = ttir.empty() : tensor<1x24x640x640xf32>
    %586 = "ttir.broadcast"(%584, %585) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %587 = ttir.empty() : tensor<1x24x640x640xf32>
    %588 = "ttir.div"(%580, %586, %587) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %589 = ttir.empty() : tensor<1x24x640x640xbf16>
    %590 = "ttir.typecast"(%588, %589) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %591 = ttir.empty() : tensor<24x640x640xbf16>
    %592 = "ttir.reshape"(%590, %591) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %593 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %594 = "ttir.reshape"(%469, %593) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %595 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %596 = "ttir.broadcast"(%594, %595) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %597 = ttir.empty() : tensor<24x640x128xbf16>
    %598 = "ttir.reshape"(%596, %597) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %599 = "ttir.dot_general"(%592, %598) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %600 = ttir.empty() : tensor<1x24x640x128xbf16>
    %601 = "ttir.reshape"(%599, %600) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %602 = ttir.empty() : tensor<1x640x24x128xbf16>
    %603 = "ttir.permute"(%601, %602) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %604 = ttir.empty() : tensor<640x3072xbf16>
    %605 = "ttir.reshape"(%603, %604) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %606 = ttir.empty() : tensor<1x3072x3072xbf16>
    %607 = "ttir.reshape"(%arg21, %606) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %608 = ttir.empty() : tensor<3072x3072xbf16>
    %609 = "ttir.reshape"(%607, %608) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %610 = ttir.empty() : tensor<3072x3072xbf16>
    %611 = "ttir.permute"(%609, %610) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %612 = "ttir.dot_general"(%605, %611) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %613 = ttir.empty() : tensor<1x640x3072xbf16>
    %614 = "ttir.reshape"(%612, %613) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %615 = ttir.empty() : tensor<1x640x3072xbf16>
    %616 = "ttir.add"(%426, %614, %615) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %617 = ttir.empty() : tensor<1x1x3072xbf16>
    %618 = "ttir.reshape"(%arg24, %617) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %619 = ttir.empty() : tensor<3072xbf16>
    %620 = "ttir.reshape"(%618, %619) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %621 = ttir.empty() : tensor<3072xf32>
    %622 = "ttir.typecast"(%620, %621) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %623 = ttir.empty() : tensor<1x1x3072xf32>
    %624 = "ttir.reshape"(%622, %623) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %625 = ttir.empty() : tensor<1x640x3072xf32>
    %626 = "ttir.broadcast"(%624, %625) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %627 = ttir.empty() : tensor<1x640x3072xf32>
    %628 = "ttir.typecast"(%616, %627) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %629 = ttir.empty() : tensor<1x640x3072xf32>
    %630 = "ttir.pow"(%628, %5, %629) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %631 = ttir.empty() : tensor<1x640xf32>
    %632 = "ttir.sum"(%630, %631) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %633 = ttir.empty() : tensor<1x640xf32>
    %634 = "ttir.multiply"(%632, %4, %633) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %635 = ttir.empty() : tensor<1x640x1xf32>
    %636 = "ttir.reshape"(%634, %635) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %637 = ttir.empty() : tensor<1x640x1xf32>
    %638 = "ttir.add"(%636, %46, %637) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %639 = ttir.empty() : tensor<1x640x1xf32>
    %640 = "ttir.rsqrt"(%638, %639) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %641 = ttir.empty() : tensor<1x640xf32>
    %642 = "ttir.reshape"(%640, %641) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %643 = ttir.empty() : tensor<1x640x1xf32>
    %644 = "ttir.reshape"(%642, %643) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %645 = ttir.empty() : tensor<1x640x3072xf32>
    %646 = "ttir.broadcast"(%644, %645) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %647 = ttir.empty() : tensor<1x640x3072xf32>
    %648 = "ttir.multiply"(%628, %646, %647) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %649 = ttir.empty() : tensor<1x640x3072xbf16>
    %650 = "ttir.typecast"(%648, %649) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %651 = ttir.empty() : tensor<1x640x3072xf32>
    %652 = "ttir.typecast"(%650, %651) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %653 = ttir.empty() : tensor<1x640x3072xf32>
    %654 = "ttir.multiply"(%626, %652, %653) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %655 = ttir.empty() : tensor<1x640x3072xbf16>
    %656 = "ttir.typecast"(%654, %655) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %657 = ttir.empty() : tensor<640x3072xbf16>
    %658 = "ttir.reshape"(%656, %657) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %659 = ttir.empty() : tensor<1x8192x3072xbf16>
    %660 = "ttir.reshape"(%arg25, %659) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %661 = ttir.empty() : tensor<8192x3072xbf16>
    %662 = "ttir.reshape"(%660, %661) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %663 = ttir.empty() : tensor<3072x8192xbf16>
    %664 = "ttir.permute"(%662, %663) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %665 = "ttir.dot_general"(%658, %664) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %666 = ttir.empty() : tensor<1x640x8192xbf16>
    %667 = "ttir.reshape"(%665, %666) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %668 = ttir.empty() : tensor<1x640x8192xf32>
    %669 = "ttir.typecast"(%667, %668) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %670 = ttir.empty() : tensor<1x640x8192xbf16>
    %671 = "ttir.sigmoid"(%667, %670) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %672 = ttir.empty() : tensor<1x640x8192xf32>
    %673 = "ttir.typecast"(%671, %672) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %674 = ttir.empty() : tensor<1x640x8192xf32>
    %675 = "ttir.multiply"(%669, %673, %674) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %676 = ttir.empty() : tensor<1x640x8192xbf16>
    %677 = "ttir.typecast"(%675, %676) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %678 = ttir.empty() : tensor<1x640x8192xf32>
    %679 = "ttir.typecast"(%677, %678) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %680 = ttir.empty() : tensor<1x8192x3072xbf16>
    %681 = "ttir.reshape"(%arg20, %680) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %682 = ttir.empty() : tensor<8192x3072xbf16>
    %683 = "ttir.reshape"(%681, %682) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %684 = ttir.empty() : tensor<3072x8192xbf16>
    %685 = "ttir.permute"(%683, %684) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %686 = "ttir.dot_general"(%658, %685) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %687 = ttir.empty() : tensor<1x640x8192xbf16>
    %688 = "ttir.reshape"(%686, %687) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %689 = ttir.empty() : tensor<1x640x8192xf32>
    %690 = "ttir.typecast"(%688, %689) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %691 = ttir.empty() : tensor<1x640x8192xf32>
    %692 = "ttir.multiply"(%679, %690, %691) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %693 = ttir.empty() : tensor<1x640x8192xbf16>
    %694 = "ttir.typecast"(%692, %693) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %695 = ttir.empty() : tensor<640x8192xbf16>
    %696 = "ttir.reshape"(%694, %695) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %697 = ttir.empty() : tensor<1x3072x8192xbf16>
    %698 = "ttir.reshape"(%arg19, %697) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %699 = ttir.empty() : tensor<3072x8192xbf16>
    %700 = "ttir.reshape"(%698, %699) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %701 = ttir.empty() : tensor<8192x3072xbf16>
    %702 = "ttir.permute"(%700, %701) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %703 = "ttir.dot_general"(%696, %702) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %704 = ttir.empty() : tensor<1x640x3072xbf16>
    %705 = "ttir.reshape"(%703, %704) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %706 = ttir.empty() : tensor<1x640x3072xbf16>
    %707 = "ttir.add"(%616, %705, %706) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %708 = ttir.empty() : tensor<1x640x3072xf32>
    %709 = "ttir.typecast"(%707, %708) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %710 = ttir.empty() : tensor<1x640x3072xf32>
    %711 = "ttir.pow"(%709, %5, %710) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %712 = ttir.empty() : tensor<1x640xf32>
    %713 = "ttir.sum"(%711, %712) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %714 = ttir.empty() : tensor<1x640xf32>
    %715 = "ttir.multiply"(%713, %4, %714) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %716 = ttir.empty() : tensor<1x640x1xf32>
    %717 = "ttir.reshape"(%715, %716) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %718 = ttir.empty() : tensor<1x640x1xf32>
    %719 = "ttir.add"(%717, %46, %718) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %720 = ttir.empty() : tensor<1x640x1xf32>
    %721 = "ttir.rsqrt"(%719, %720) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %722 = ttir.empty() : tensor<1x640xf32>
    %723 = "ttir.reshape"(%721, %722) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %724 = ttir.empty() : tensor<1x640x1xf32>
    %725 = "ttir.reshape"(%723, %724) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %726 = ttir.empty() : tensor<1x640x3072xf32>
    %727 = "ttir.broadcast"(%725, %726) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %728 = ttir.empty() : tensor<1x640x3072xf32>
    %729 = "ttir.multiply"(%709, %727, %728) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %730 = ttir.empty() : tensor<1x640x3072xbf16>
    %731 = "ttir.typecast"(%729, %730) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %732 = ttir.empty() : tensor<1x640x3072xf32>
    %733 = "ttir.typecast"(%731, %732) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %734 = ttir.empty() : tensor<1x640x3072xf32>
    %735 = "ttir.multiply"(%479, %733, %734) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %736 = ttir.empty() : tensor<1x640x3072xbf16>
    %737 = "ttir.typecast"(%735, %736) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %738 = ttir.empty() : tensor<640x3072xbf16>
    %739 = "ttir.reshape"(%737, %738) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %740 = ttir.empty() : tensor<1x1024x3072xbf16>
    %741 = "ttir.reshape"(%arg18, %740) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %742 = ttir.empty() : tensor<1024x3072xbf16>
    %743 = "ttir.reshape"(%741, %742) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %744 = ttir.empty() : tensor<3072x1024xbf16>
    %745 = "ttir.permute"(%743, %744) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %746 = "ttir.dot_general"(%739, %745) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %747 = ttir.empty() : tensor<1x640x8x128xbf16>
    %748 = "ttir.reshape"(%746, %747) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %749 = ttir.empty() : tensor<1x8x640x128xbf16>
    %750 = "ttir.permute"(%748, %749) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %751 = ttir.empty() : tensor<1x1x3072xbf16>
    %752 = "ttir.reshape"(%arg35, %751) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %753 = ttir.empty() : tensor<3072xbf16>
    %754 = "ttir.reshape"(%752, %753) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %755 = ttir.empty() : tensor<3072xf32>
    %756 = "ttir.typecast"(%754, %755) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %757 = ttir.empty() : tensor<1x1x3072xf32>
    %758 = "ttir.reshape"(%756, %757) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %759 = ttir.empty() : tensor<1x640x3072xf32>
    %760 = "ttir.broadcast"(%758, %759) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %761 = ttir.empty() : tensor<1x3072x3072xbf16>
    %762 = "ttir.reshape"(%arg32, %761) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %763 = ttir.empty() : tensor<3072x3072xbf16>
    %764 = "ttir.reshape"(%762, %763) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %765 = ttir.empty() : tensor<3072x3072xbf16>
    %766 = "ttir.permute"(%764, %765) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %767 = "ttir.dot_general"(%739, %766) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %768 = ttir.empty() : tensor<1x640x24x128xbf16>
    %769 = "ttir.reshape"(%767, %768) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %770 = ttir.empty() : tensor<1x24x640x128xbf16>
    %771 = "ttir.permute"(%769, %770) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %772 = ttir.empty() : tensor<1x24x640x128xf32>
    %773 = "ttir.typecast"(%771, %772) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %774 = ttir.empty() : tensor<1x24x640x128xf32>
    %775 = "ttir.multiply"(%773, %125, %774) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %776 = ttir.empty() : tensor<1x24x640x128xbf16>
    %777 = "ttir.typecast"(%775, %776) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %778 = ttir.empty() : tensor<1x24x640x64xbf16>
    %779 = "ttir.slice_static"(%771, %778) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %780 = ttir.empty() : tensor<1x24x640x64xbf16>
    %781 = "ttir.neg"(%779, %780) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %782 = ttir.empty() : tensor<1x24x640x64xbf16>
    %783 = "ttir.slice_static"(%771, %782) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %784 = ttir.empty() : tensor<1x24x640x128xbf16>
    %785 = "ttir.concat"(%781, %783, %784) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %786 = ttir.empty() : tensor<1x24x640x128xf32>
    %787 = "ttir.typecast"(%785, %786) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %788 = ttir.empty() : tensor<1x24x640x128xf32>
    %789 = "ttir.multiply"(%787, %153, %788) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %790 = ttir.empty() : tensor<1x24x640x128xbf16>
    %791 = "ttir.typecast"(%789, %790) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %792 = ttir.empty() : tensor<1x24x640x128xbf16>
    %793 = "ttir.add"(%777, %791, %792) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %794 = ttir.empty() : tensor<24x640x128xbf16>
    %795 = "ttir.reshape"(%793, %794) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %796 = ttir.empty() : tensor<1x1024x3072xbf16>
    %797 = "ttir.reshape"(%arg31, %796) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %798 = ttir.empty() : tensor<1024x3072xbf16>
    %799 = "ttir.reshape"(%797, %798) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %800 = ttir.empty() : tensor<3072x1024xbf16>
    %801 = "ttir.permute"(%799, %800) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %802 = "ttir.dot_general"(%739, %801) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %803 = ttir.empty() : tensor<1x640x8x128xbf16>
    %804 = "ttir.reshape"(%802, %803) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %805 = ttir.empty() : tensor<1x8x640x128xbf16>
    %806 = "ttir.permute"(%804, %805) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %807 = ttir.empty() : tensor<1x8x640x128xf32>
    %808 = "ttir.typecast"(%806, %807) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %809 = ttir.empty() : tensor<1x8x640x128xf32>
    %810 = "ttir.multiply"(%808, %178, %809) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %811 = ttir.empty() : tensor<1x8x640x128xbf16>
    %812 = "ttir.typecast"(%810, %811) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %813 = ttir.empty() : tensor<1x8x640x64xbf16>
    %814 = "ttir.slice_static"(%806, %813) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %815 = ttir.empty() : tensor<1x8x640x64xbf16>
    %816 = "ttir.neg"(%814, %815) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %817 = ttir.empty() : tensor<1x8x640x64xbf16>
    %818 = "ttir.slice_static"(%806, %817) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %819 = ttir.empty() : tensor<1x8x640x128xbf16>
    %820 = "ttir.concat"(%816, %818, %819) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %821 = ttir.empty() : tensor<1x8x640x128xf32>
    %822 = "ttir.typecast"(%820, %821) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %823 = ttir.empty() : tensor<1x8x640x128xf32>
    %824 = "ttir.multiply"(%822, %196, %823) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %825 = ttir.empty() : tensor<1x8x640x128xbf16>
    %826 = "ttir.typecast"(%824, %825) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %827 = ttir.empty() : tensor<1x8x640x128xbf16>
    %828 = "ttir.add"(%812, %826, %827) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %829 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %830 = "ttir.reshape"(%828, %829) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %831 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %832 = "ttir.broadcast"(%830, %831) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %833 = ttir.empty() : tensor<1x24x640x128xbf16>
    %834 = "ttir.reshape"(%832, %833) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %835 = ttir.empty() : tensor<1x24x128x640xbf16>
    %836 = "ttir.permute"(%834, %835) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %837 = ttir.empty() : tensor<24x128x640xbf16>
    %838 = "ttir.reshape"(%836, %837) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %839 = "ttir.dot_general"(%795, %838) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %840 = ttir.empty() : tensor<1x24x640x640xbf16>
    %841 = "ttir.reshape"(%839, %840) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %842 = ttir.empty() : tensor<1x24x640x640xf32>
    %843 = "ttir.typecast"(%841, %842) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %844 = ttir.empty() : tensor<1x24x640x640xf32>
    %845 = "ttir.multiply"(%843, %221, %844) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %846 = ttir.empty() : tensor<1x24x640x640xbf16>
    %847 = "ttir.typecast"(%845, %846) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %848 = ttir.empty() : tensor<1x24x640x640xbf16>
    %849 = "ttir.add"(%847, %285, %848) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %850 = ttir.empty() : tensor<1x24x640x640xf32>
    %851 = "ttir.typecast"(%849, %850) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %852 = ttir.empty() : tensor<1x24x640xf32>
    %853 = "ttir.max"(%851, %852) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %854 = ttir.empty() : tensor<1x24x640x1xf32>
    %855 = "ttir.reshape"(%853, %854) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %856 = ttir.empty() : tensor<1x24x640x640xf32>
    %857 = "ttir.broadcast"(%855, %856) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %858 = ttir.empty() : tensor<1x24x640x640xf32>
    %859 = "ttir.subtract"(%851, %857, %858) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %860 = ttir.empty() : tensor<1x24x640x640xf32>
    %861 = "ttir.exp"(%859, %860) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %862 = ttir.empty() : tensor<1x24x640xf32>
    %863 = "ttir.sum"(%861, %862) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %864 = ttir.empty() : tensor<1x24x640x1xf32>
    %865 = "ttir.reshape"(%863, %864) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %866 = ttir.empty() : tensor<1x24x640x640xf32>
    %867 = "ttir.broadcast"(%865, %866) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %868 = ttir.empty() : tensor<1x24x640x640xf32>
    %869 = "ttir.div"(%861, %867, %868) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %870 = ttir.empty() : tensor<1x24x640x640xbf16>
    %871 = "ttir.typecast"(%869, %870) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %872 = ttir.empty() : tensor<24x640x640xbf16>
    %873 = "ttir.reshape"(%871, %872) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %874 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %875 = "ttir.reshape"(%750, %874) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %876 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %877 = "ttir.broadcast"(%875, %876) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %878 = ttir.empty() : tensor<24x640x128xbf16>
    %879 = "ttir.reshape"(%877, %878) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %880 = "ttir.dot_general"(%873, %879) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %881 = ttir.empty() : tensor<1x24x640x128xbf16>
    %882 = "ttir.reshape"(%880, %881) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %883 = ttir.empty() : tensor<1x640x24x128xbf16>
    %884 = "ttir.permute"(%882, %883) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %885 = ttir.empty() : tensor<640x3072xbf16>
    %886 = "ttir.reshape"(%884, %885) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %887 = ttir.empty() : tensor<1x3072x3072xbf16>
    %888 = "ttir.reshape"(%arg30, %887) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %889 = ttir.empty() : tensor<3072x3072xbf16>
    %890 = "ttir.reshape"(%888, %889) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %891 = ttir.empty() : tensor<3072x3072xbf16>
    %892 = "ttir.permute"(%890, %891) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %893 = "ttir.dot_general"(%886, %892) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %894 = ttir.empty() : tensor<1x640x3072xbf16>
    %895 = "ttir.reshape"(%893, %894) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %896 = ttir.empty() : tensor<1x640x3072xbf16>
    %897 = "ttir.add"(%707, %895, %896) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %898 = ttir.empty() : tensor<1x1x3072xbf16>
    %899 = "ttir.reshape"(%arg33, %898) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %900 = ttir.empty() : tensor<3072xbf16>
    %901 = "ttir.reshape"(%899, %900) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %902 = ttir.empty() : tensor<3072xf32>
    %903 = "ttir.typecast"(%901, %902) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %904 = ttir.empty() : tensor<1x1x3072xf32>
    %905 = "ttir.reshape"(%903, %904) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %906 = ttir.empty() : tensor<1x640x3072xf32>
    %907 = "ttir.broadcast"(%905, %906) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %908 = ttir.empty() : tensor<1x640x3072xf32>
    %909 = "ttir.typecast"(%897, %908) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %910 = ttir.empty() : tensor<1x640x3072xf32>
    %911 = "ttir.pow"(%909, %5, %910) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %912 = ttir.empty() : tensor<1x640xf32>
    %913 = "ttir.sum"(%911, %912) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %914 = ttir.empty() : tensor<1x640xf32>
    %915 = "ttir.multiply"(%913, %4, %914) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %916 = ttir.empty() : tensor<1x640x1xf32>
    %917 = "ttir.reshape"(%915, %916) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %918 = ttir.empty() : tensor<1x640x1xf32>
    %919 = "ttir.add"(%917, %46, %918) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %920 = ttir.empty() : tensor<1x640x1xf32>
    %921 = "ttir.rsqrt"(%919, %920) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %922 = ttir.empty() : tensor<1x640xf32>
    %923 = "ttir.reshape"(%921, %922) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %924 = ttir.empty() : tensor<1x640x1xf32>
    %925 = "ttir.reshape"(%923, %924) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %926 = ttir.empty() : tensor<1x640x3072xf32>
    %927 = "ttir.broadcast"(%925, %926) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %928 = ttir.empty() : tensor<1x640x3072xf32>
    %929 = "ttir.multiply"(%909, %927, %928) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %930 = ttir.empty() : tensor<1x640x3072xbf16>
    %931 = "ttir.typecast"(%929, %930) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %932 = ttir.empty() : tensor<1x640x3072xf32>
    %933 = "ttir.typecast"(%931, %932) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %934 = ttir.empty() : tensor<1x640x3072xf32>
    %935 = "ttir.multiply"(%907, %933, %934) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %936 = ttir.empty() : tensor<1x640x3072xbf16>
    %937 = "ttir.typecast"(%935, %936) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %938 = ttir.empty() : tensor<640x3072xbf16>
    %939 = "ttir.reshape"(%937, %938) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %940 = ttir.empty() : tensor<1x8192x3072xbf16>
    %941 = "ttir.reshape"(%arg34, %940) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %942 = ttir.empty() : tensor<8192x3072xbf16>
    %943 = "ttir.reshape"(%941, %942) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %944 = ttir.empty() : tensor<3072x8192xbf16>
    %945 = "ttir.permute"(%943, %944) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %946 = "ttir.dot_general"(%939, %945) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %947 = ttir.empty() : tensor<1x640x8192xbf16>
    %948 = "ttir.reshape"(%946, %947) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %949 = ttir.empty() : tensor<1x640x8192xf32>
    %950 = "ttir.typecast"(%948, %949) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %951 = ttir.empty() : tensor<1x640x8192xbf16>
    %952 = "ttir.sigmoid"(%948, %951) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %953 = ttir.empty() : tensor<1x640x8192xf32>
    %954 = "ttir.typecast"(%952, %953) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %955 = ttir.empty() : tensor<1x640x8192xf32>
    %956 = "ttir.multiply"(%950, %954, %955) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %957 = ttir.empty() : tensor<1x640x8192xbf16>
    %958 = "ttir.typecast"(%956, %957) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %959 = ttir.empty() : tensor<1x640x8192xf32>
    %960 = "ttir.typecast"(%958, %959) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %961 = ttir.empty() : tensor<1x8192x3072xbf16>
    %962 = "ttir.reshape"(%arg29, %961) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %963 = ttir.empty() : tensor<8192x3072xbf16>
    %964 = "ttir.reshape"(%962, %963) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %965 = ttir.empty() : tensor<3072x8192xbf16>
    %966 = "ttir.permute"(%964, %965) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %967 = "ttir.dot_general"(%939, %966) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %968 = ttir.empty() : tensor<1x640x8192xbf16>
    %969 = "ttir.reshape"(%967, %968) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %970 = ttir.empty() : tensor<1x640x8192xf32>
    %971 = "ttir.typecast"(%969, %970) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %972 = ttir.empty() : tensor<1x640x8192xf32>
    %973 = "ttir.multiply"(%960, %971, %972) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %974 = ttir.empty() : tensor<1x640x8192xbf16>
    %975 = "ttir.typecast"(%973, %974) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %976 = ttir.empty() : tensor<640x8192xbf16>
    %977 = "ttir.reshape"(%975, %976) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %978 = ttir.empty() : tensor<1x3072x8192xbf16>
    %979 = "ttir.reshape"(%arg28, %978) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %980 = ttir.empty() : tensor<3072x8192xbf16>
    %981 = "ttir.reshape"(%979, %980) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %982 = ttir.empty() : tensor<8192x3072xbf16>
    %983 = "ttir.permute"(%981, %982) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %984 = "ttir.dot_general"(%977, %983) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %985 = ttir.empty() : tensor<1x640x3072xbf16>
    %986 = "ttir.reshape"(%984, %985) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %987 = ttir.empty() : tensor<1x640x3072xbf16>
    %988 = "ttir.add"(%897, %986, %987) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %989 = ttir.empty() : tensor<1x640x3072xf32>
    %990 = "ttir.typecast"(%988, %989) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %991 = ttir.empty() : tensor<1x640x3072xf32>
    %992 = "ttir.pow"(%990, %5, %991) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %993 = ttir.empty() : tensor<1x640xf32>
    %994 = "ttir.sum"(%992, %993) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %995 = ttir.empty() : tensor<1x640xf32>
    %996 = "ttir.multiply"(%994, %4, %995) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %997 = ttir.empty() : tensor<1x640x1xf32>
    %998 = "ttir.reshape"(%996, %997) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %999 = ttir.empty() : tensor<1x640x1xf32>
    %1000 = "ttir.add"(%998, %46, %999) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1001 = ttir.empty() : tensor<1x640x1xf32>
    %1002 = "ttir.rsqrt"(%1000, %1001) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1003 = ttir.empty() : tensor<1x640xf32>
    %1004 = "ttir.reshape"(%1002, %1003) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1005 = ttir.empty() : tensor<1x640x1xf32>
    %1006 = "ttir.reshape"(%1004, %1005) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1007 = ttir.empty() : tensor<1x640x3072xf32>
    %1008 = "ttir.broadcast"(%1006, %1007) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1009 = ttir.empty() : tensor<1x640x3072xf32>
    %1010 = "ttir.multiply"(%990, %1008, %1009) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1011 = ttir.empty() : tensor<1x640x3072xbf16>
    %1012 = "ttir.typecast"(%1010, %1011) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1013 = ttir.empty() : tensor<1x640x3072xf32>
    %1014 = "ttir.typecast"(%1012, %1013) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1015 = ttir.empty() : tensor<1x640x3072xf32>
    %1016 = "ttir.multiply"(%760, %1014, %1015) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1017 = ttir.empty() : tensor<1x640x3072xbf16>
    %1018 = "ttir.typecast"(%1016, %1017) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1019 = ttir.empty() : tensor<640x3072xbf16>
    %1020 = "ttir.reshape"(%1018, %1019) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %1021 = ttir.empty() : tensor<1x1024x3072xbf16>
    %1022 = "ttir.reshape"(%arg27, %1021) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %1023 = ttir.empty() : tensor<1024x3072xbf16>
    %1024 = "ttir.reshape"(%1022, %1023) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %1025 = ttir.empty() : tensor<3072x1024xbf16>
    %1026 = "ttir.permute"(%1024, %1025) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %1027 = "ttir.dot_general"(%1020, %1026) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %1028 = ttir.empty() : tensor<1x640x8x128xbf16>
    %1029 = "ttir.reshape"(%1027, %1028) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %1030 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1031 = "ttir.permute"(%1029, %1030) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1032 = ttir.empty() : tensor<1x1x3072xbf16>
    %1033 = "ttir.reshape"(%arg44, %1032) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %1034 = ttir.empty() : tensor<3072xbf16>
    %1035 = "ttir.reshape"(%1033, %1034) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %1036 = ttir.empty() : tensor<3072xf32>
    %1037 = "ttir.typecast"(%1035, %1036) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %1038 = ttir.empty() : tensor<1x1x3072xf32>
    %1039 = "ttir.reshape"(%1037, %1038) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %1040 = ttir.empty() : tensor<1x640x3072xf32>
    %1041 = "ttir.broadcast"(%1039, %1040) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1042 = ttir.empty() : tensor<1x3072x3072xbf16>
    %1043 = "ttir.reshape"(%arg41, %1042) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %1044 = ttir.empty() : tensor<3072x3072xbf16>
    %1045 = "ttir.reshape"(%1043, %1044) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1046 = ttir.empty() : tensor<3072x3072xbf16>
    %1047 = "ttir.permute"(%1045, %1046) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1048 = "ttir.dot_general"(%1020, %1047) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %1049 = ttir.empty() : tensor<1x640x24x128xbf16>
    %1050 = "ttir.reshape"(%1048, %1049) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %1051 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1052 = "ttir.permute"(%1050, %1051) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1053 = ttir.empty() : tensor<1x24x640x128xf32>
    %1054 = "ttir.typecast"(%1052, %1053) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1055 = ttir.empty() : tensor<1x24x640x128xf32>
    %1056 = "ttir.multiply"(%1054, %125, %1055) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1057 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1058 = "ttir.typecast"(%1056, %1057) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1059 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1060 = "ttir.slice_static"(%1052, %1059) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1061 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1062 = "ttir.neg"(%1060, %1061) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1063 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1064 = "ttir.slice_static"(%1052, %1063) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1065 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1066 = "ttir.concat"(%1062, %1064, %1065) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1067 = ttir.empty() : tensor<1x24x640x128xf32>
    %1068 = "ttir.typecast"(%1066, %1067) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1069 = ttir.empty() : tensor<1x24x640x128xf32>
    %1070 = "ttir.multiply"(%1068, %153, %1069) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1071 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1072 = "ttir.typecast"(%1070, %1071) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1073 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1074 = "ttir.add"(%1058, %1072, %1073) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1075 = ttir.empty() : tensor<24x640x128xbf16>
    %1076 = "ttir.reshape"(%1074, %1075) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %1077 = ttir.empty() : tensor<1x1024x3072xbf16>
    %1078 = "ttir.reshape"(%arg40, %1077) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %1079 = ttir.empty() : tensor<1024x3072xbf16>
    %1080 = "ttir.reshape"(%1078, %1079) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %1081 = ttir.empty() : tensor<3072x1024xbf16>
    %1082 = "ttir.permute"(%1080, %1081) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %1083 = "ttir.dot_general"(%1020, %1082) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %1084 = ttir.empty() : tensor<1x640x8x128xbf16>
    %1085 = "ttir.reshape"(%1083, %1084) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %1086 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1087 = "ttir.permute"(%1085, %1086) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1088 = ttir.empty() : tensor<1x8x640x128xf32>
    %1089 = "ttir.typecast"(%1087, %1088) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1090 = ttir.empty() : tensor<1x8x640x128xf32>
    %1091 = "ttir.multiply"(%1089, %178, %1090) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1092 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1093 = "ttir.typecast"(%1091, %1092) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1094 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1095 = "ttir.slice_static"(%1087, %1094) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1096 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1097 = "ttir.neg"(%1095, %1096) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1098 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1099 = "ttir.slice_static"(%1087, %1098) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1100 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1101 = "ttir.concat"(%1097, %1099, %1100) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1102 = ttir.empty() : tensor<1x8x640x128xf32>
    %1103 = "ttir.typecast"(%1101, %1102) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1104 = ttir.empty() : tensor<1x8x640x128xf32>
    %1105 = "ttir.multiply"(%1103, %196, %1104) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1106 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1107 = "ttir.typecast"(%1105, %1106) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1108 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1109 = "ttir.add"(%1093, %1107, %1108) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1110 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %1111 = "ttir.reshape"(%1109, %1110) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %1112 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %1113 = "ttir.broadcast"(%1111, %1112) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %1114 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1115 = "ttir.reshape"(%1113, %1114) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1116 = ttir.empty() : tensor<1x24x128x640xbf16>
    %1117 = "ttir.permute"(%1115, %1116) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %1118 = ttir.empty() : tensor<24x128x640xbf16>
    %1119 = "ttir.reshape"(%1117, %1118) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %1120 = "ttir.dot_general"(%1076, %1119) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %1121 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1122 = "ttir.reshape"(%1120, %1121) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1123 = ttir.empty() : tensor<1x24x640x640xf32>
    %1124 = "ttir.typecast"(%1122, %1123) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1125 = ttir.empty() : tensor<1x24x640x640xf32>
    %1126 = "ttir.multiply"(%1124, %221, %1125) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1127 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1128 = "ttir.typecast"(%1126, %1127) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1129 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1130 = "ttir.add"(%1128, %285, %1129) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1131 = ttir.empty() : tensor<1x24x640x640xf32>
    %1132 = "ttir.typecast"(%1130, %1131) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1133 = ttir.empty() : tensor<1x24x640xf32>
    %1134 = "ttir.max"(%1132, %1133) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %1135 = ttir.empty() : tensor<1x24x640x1xf32>
    %1136 = "ttir.reshape"(%1134, %1135) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %1137 = ttir.empty() : tensor<1x24x640x640xf32>
    %1138 = "ttir.broadcast"(%1136, %1137) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1139 = ttir.empty() : tensor<1x24x640x640xf32>
    %1140 = "ttir.subtract"(%1132, %1138, %1139) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1141 = ttir.empty() : tensor<1x24x640x640xf32>
    %1142 = "ttir.exp"(%1140, %1141) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1143 = ttir.empty() : tensor<1x24x640xf32>
    %1144 = "ttir.sum"(%1142, %1143) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %1145 = ttir.empty() : tensor<1x24x640x1xf32>
    %1146 = "ttir.reshape"(%1144, %1145) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %1147 = ttir.empty() : tensor<1x24x640x640xf32>
    %1148 = "ttir.broadcast"(%1146, %1147) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1149 = ttir.empty() : tensor<1x24x640x640xf32>
    %1150 = "ttir.div"(%1142, %1148, %1149) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1151 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1152 = "ttir.typecast"(%1150, %1151) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1153 = ttir.empty() : tensor<24x640x640xbf16>
    %1154 = "ttir.reshape"(%1152, %1153) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %1155 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %1156 = "ttir.reshape"(%1031, %1155) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %1157 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %1158 = "ttir.broadcast"(%1156, %1157) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %1159 = ttir.empty() : tensor<24x640x128xbf16>
    %1160 = "ttir.reshape"(%1158, %1159) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %1161 = "ttir.dot_general"(%1154, %1160) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %1162 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1163 = "ttir.reshape"(%1161, %1162) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1164 = ttir.empty() : tensor<1x640x24x128xbf16>
    %1165 = "ttir.permute"(%1163, %1164) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %1166 = ttir.empty() : tensor<640x3072xbf16>
    %1167 = "ttir.reshape"(%1165, %1166) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %1168 = ttir.empty() : tensor<1x3072x3072xbf16>
    %1169 = "ttir.reshape"(%arg39, %1168) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %1170 = ttir.empty() : tensor<3072x3072xbf16>
    %1171 = "ttir.reshape"(%1169, %1170) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1172 = ttir.empty() : tensor<3072x3072xbf16>
    %1173 = "ttir.permute"(%1171, %1172) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1174 = "ttir.dot_general"(%1167, %1173) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %1175 = ttir.empty() : tensor<1x640x3072xbf16>
    %1176 = "ttir.reshape"(%1174, %1175) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1177 = ttir.empty() : tensor<1x640x3072xbf16>
    %1178 = "ttir.add"(%988, %1176, %1177) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1179 = ttir.empty() : tensor<1x1x3072xbf16>
    %1180 = "ttir.reshape"(%arg42, %1179) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %1181 = ttir.empty() : tensor<3072xbf16>
    %1182 = "ttir.reshape"(%1180, %1181) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %1183 = ttir.empty() : tensor<3072xf32>
    %1184 = "ttir.typecast"(%1182, %1183) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %1185 = ttir.empty() : tensor<1x1x3072xf32>
    %1186 = "ttir.reshape"(%1184, %1185) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %1187 = ttir.empty() : tensor<1x640x3072xf32>
    %1188 = "ttir.broadcast"(%1186, %1187) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1189 = ttir.empty() : tensor<1x640x3072xf32>
    %1190 = "ttir.typecast"(%1178, %1189) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1191 = ttir.empty() : tensor<1x640x3072xf32>
    %1192 = "ttir.pow"(%1190, %5, %1191) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1193 = ttir.empty() : tensor<1x640xf32>
    %1194 = "ttir.sum"(%1192, %1193) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1195 = ttir.empty() : tensor<1x640xf32>
    %1196 = "ttir.multiply"(%1194, %4, %1195) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1197 = ttir.empty() : tensor<1x640x1xf32>
    %1198 = "ttir.reshape"(%1196, %1197) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1199 = ttir.empty() : tensor<1x640x1xf32>
    %1200 = "ttir.add"(%1198, %46, %1199) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1201 = ttir.empty() : tensor<1x640x1xf32>
    %1202 = "ttir.rsqrt"(%1200, %1201) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1203 = ttir.empty() : tensor<1x640xf32>
    %1204 = "ttir.reshape"(%1202, %1203) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1205 = ttir.empty() : tensor<1x640x1xf32>
    %1206 = "ttir.reshape"(%1204, %1205) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1207 = ttir.empty() : tensor<1x640x3072xf32>
    %1208 = "ttir.broadcast"(%1206, %1207) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1209 = ttir.empty() : tensor<1x640x3072xf32>
    %1210 = "ttir.multiply"(%1190, %1208, %1209) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1211 = ttir.empty() : tensor<1x640x3072xbf16>
    %1212 = "ttir.typecast"(%1210, %1211) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1213 = ttir.empty() : tensor<1x640x3072xf32>
    %1214 = "ttir.typecast"(%1212, %1213) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1215 = ttir.empty() : tensor<1x640x3072xf32>
    %1216 = "ttir.multiply"(%1188, %1214, %1215) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1217 = ttir.empty() : tensor<1x640x3072xbf16>
    %1218 = "ttir.typecast"(%1216, %1217) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1219 = ttir.empty() : tensor<640x3072xbf16>
    %1220 = "ttir.reshape"(%1218, %1219) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %1221 = ttir.empty() : tensor<1x8192x3072xbf16>
    %1222 = "ttir.reshape"(%arg43, %1221) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %1223 = ttir.empty() : tensor<8192x3072xbf16>
    %1224 = "ttir.reshape"(%1222, %1223) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %1225 = ttir.empty() : tensor<3072x8192xbf16>
    %1226 = "ttir.permute"(%1224, %1225) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %1227 = "ttir.dot_general"(%1220, %1226) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %1228 = ttir.empty() : tensor<1x640x8192xbf16>
    %1229 = "ttir.reshape"(%1227, %1228) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1230 = ttir.empty() : tensor<1x640x8192xf32>
    %1231 = "ttir.typecast"(%1229, %1230) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1232 = ttir.empty() : tensor<1x640x8192xbf16>
    %1233 = "ttir.sigmoid"(%1229, %1232) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1234 = ttir.empty() : tensor<1x640x8192xf32>
    %1235 = "ttir.typecast"(%1233, %1234) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1236 = ttir.empty() : tensor<1x640x8192xf32>
    %1237 = "ttir.multiply"(%1231, %1235, %1236) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1238 = ttir.empty() : tensor<1x640x8192xbf16>
    %1239 = "ttir.typecast"(%1237, %1238) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1240 = ttir.empty() : tensor<1x640x8192xf32>
    %1241 = "ttir.typecast"(%1239, %1240) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1242 = ttir.empty() : tensor<1x8192x3072xbf16>
    %1243 = "ttir.reshape"(%arg38, %1242) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %1244 = ttir.empty() : tensor<8192x3072xbf16>
    %1245 = "ttir.reshape"(%1243, %1244) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %1246 = ttir.empty() : tensor<3072x8192xbf16>
    %1247 = "ttir.permute"(%1245, %1246) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %1248 = "ttir.dot_general"(%1220, %1247) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %1249 = ttir.empty() : tensor<1x640x8192xbf16>
    %1250 = "ttir.reshape"(%1248, %1249) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1251 = ttir.empty() : tensor<1x640x8192xf32>
    %1252 = "ttir.typecast"(%1250, %1251) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1253 = ttir.empty() : tensor<1x640x8192xf32>
    %1254 = "ttir.multiply"(%1241, %1252, %1253) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1255 = ttir.empty() : tensor<1x640x8192xbf16>
    %1256 = "ttir.typecast"(%1254, %1255) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1257 = ttir.empty() : tensor<640x8192xbf16>
    %1258 = "ttir.reshape"(%1256, %1257) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %1259 = ttir.empty() : tensor<1x3072x8192xbf16>
    %1260 = "ttir.reshape"(%arg37, %1259) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %1261 = ttir.empty() : tensor<3072x8192xbf16>
    %1262 = "ttir.reshape"(%1260, %1261) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %1263 = ttir.empty() : tensor<8192x3072xbf16>
    %1264 = "ttir.permute"(%1262, %1263) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %1265 = "ttir.dot_general"(%1258, %1264) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %1266 = ttir.empty() : tensor<1x640x3072xbf16>
    %1267 = "ttir.reshape"(%1265, %1266) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1268 = ttir.empty() : tensor<1x640x3072xbf16>
    %1269 = "ttir.add"(%1178, %1267, %1268) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1270 = ttir.empty() : tensor<1x640x3072xf32>
    %1271 = "ttir.typecast"(%1269, %1270) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1272 = ttir.empty() : tensor<1x640x3072xf32>
    %1273 = "ttir.pow"(%1271, %5, %1272) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1274 = ttir.empty() : tensor<1x640xf32>
    %1275 = "ttir.sum"(%1273, %1274) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1276 = ttir.empty() : tensor<1x640xf32>
    %1277 = "ttir.multiply"(%1275, %4, %1276) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1278 = ttir.empty() : tensor<1x640x1xf32>
    %1279 = "ttir.reshape"(%1277, %1278) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1280 = ttir.empty() : tensor<1x640x1xf32>
    %1281 = "ttir.add"(%1279, %46, %1280) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1282 = ttir.empty() : tensor<1x640x1xf32>
    %1283 = "ttir.rsqrt"(%1281, %1282) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1284 = ttir.empty() : tensor<1x640xf32>
    %1285 = "ttir.reshape"(%1283, %1284) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1286 = ttir.empty() : tensor<1x640x1xf32>
    %1287 = "ttir.reshape"(%1285, %1286) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1288 = ttir.empty() : tensor<1x640x3072xf32>
    %1289 = "ttir.broadcast"(%1287, %1288) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1290 = ttir.empty() : tensor<1x640x3072xf32>
    %1291 = "ttir.multiply"(%1271, %1289, %1290) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1292 = ttir.empty() : tensor<1x640x3072xbf16>
    %1293 = "ttir.typecast"(%1291, %1292) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1294 = ttir.empty() : tensor<1x640x3072xf32>
    %1295 = "ttir.typecast"(%1293, %1294) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1296 = ttir.empty() : tensor<1x640x3072xf32>
    %1297 = "ttir.multiply"(%1041, %1295, %1296) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1298 = ttir.empty() : tensor<1x640x3072xbf16>
    %1299 = "ttir.typecast"(%1297, %1298) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1300 = ttir.empty() : tensor<640x3072xbf16>
    %1301 = "ttir.reshape"(%1299, %1300) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %1302 = ttir.empty() : tensor<1x1024x3072xbf16>
    %1303 = "ttir.reshape"(%arg36, %1302) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %1304 = ttir.empty() : tensor<1024x3072xbf16>
    %1305 = "ttir.reshape"(%1303, %1304) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %1306 = ttir.empty() : tensor<3072x1024xbf16>
    %1307 = "ttir.permute"(%1305, %1306) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %1308 = "ttir.dot_general"(%1301, %1307) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %1309 = ttir.empty() : tensor<1x640x8x128xbf16>
    %1310 = "ttir.reshape"(%1308, %1309) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %1311 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1312 = "ttir.permute"(%1310, %1311) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1313 = ttir.empty() : tensor<1x1x3072xbf16>
    %1314 = "ttir.reshape"(%arg53, %1313) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %1315 = ttir.empty() : tensor<3072xbf16>
    %1316 = "ttir.reshape"(%1314, %1315) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %1317 = ttir.empty() : tensor<3072xf32>
    %1318 = "ttir.typecast"(%1316, %1317) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %1319 = ttir.empty() : tensor<1x1x3072xf32>
    %1320 = "ttir.reshape"(%1318, %1319) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %1321 = ttir.empty() : tensor<1x640x3072xf32>
    %1322 = "ttir.broadcast"(%1320, %1321) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1323 = ttir.empty() : tensor<1x3072x3072xbf16>
    %1324 = "ttir.reshape"(%arg50, %1323) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %1325 = ttir.empty() : tensor<3072x3072xbf16>
    %1326 = "ttir.reshape"(%1324, %1325) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1327 = ttir.empty() : tensor<3072x3072xbf16>
    %1328 = "ttir.permute"(%1326, %1327) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1329 = "ttir.dot_general"(%1301, %1328) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %1330 = ttir.empty() : tensor<1x640x24x128xbf16>
    %1331 = "ttir.reshape"(%1329, %1330) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %1332 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1333 = "ttir.permute"(%1331, %1332) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1334 = ttir.empty() : tensor<1x24x640x128xf32>
    %1335 = "ttir.typecast"(%1333, %1334) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1336 = ttir.empty() : tensor<1x24x640x128xf32>
    %1337 = "ttir.multiply"(%1335, %125, %1336) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1338 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1339 = "ttir.typecast"(%1337, %1338) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1340 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1341 = "ttir.slice_static"(%1333, %1340) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1342 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1343 = "ttir.neg"(%1341, %1342) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1344 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1345 = "ttir.slice_static"(%1333, %1344) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1346 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1347 = "ttir.concat"(%1343, %1345, %1346) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1348 = ttir.empty() : tensor<1x24x640x128xf32>
    %1349 = "ttir.typecast"(%1347, %1348) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1350 = ttir.empty() : tensor<1x24x640x128xf32>
    %1351 = "ttir.multiply"(%1349, %153, %1350) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1352 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1353 = "ttir.typecast"(%1351, %1352) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1354 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1355 = "ttir.add"(%1339, %1353, %1354) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1356 = ttir.empty() : tensor<24x640x128xbf16>
    %1357 = "ttir.reshape"(%1355, %1356) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %1358 = ttir.empty() : tensor<1x1024x3072xbf16>
    %1359 = "ttir.reshape"(%arg49, %1358) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %1360 = ttir.empty() : tensor<1024x3072xbf16>
    %1361 = "ttir.reshape"(%1359, %1360) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %1362 = ttir.empty() : tensor<3072x1024xbf16>
    %1363 = "ttir.permute"(%1361, %1362) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %1364 = "ttir.dot_general"(%1301, %1363) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %1365 = ttir.empty() : tensor<1x640x8x128xbf16>
    %1366 = "ttir.reshape"(%1364, %1365) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %1367 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1368 = "ttir.permute"(%1366, %1367) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1369 = ttir.empty() : tensor<1x8x640x128xf32>
    %1370 = "ttir.typecast"(%1368, %1369) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1371 = ttir.empty() : tensor<1x8x640x128xf32>
    %1372 = "ttir.multiply"(%1370, %178, %1371) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1373 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1374 = "ttir.typecast"(%1372, %1373) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1375 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1376 = "ttir.slice_static"(%1368, %1375) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1377 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1378 = "ttir.neg"(%1376, %1377) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1379 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1380 = "ttir.slice_static"(%1368, %1379) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1381 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1382 = "ttir.concat"(%1378, %1380, %1381) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1383 = ttir.empty() : tensor<1x8x640x128xf32>
    %1384 = "ttir.typecast"(%1382, %1383) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1385 = ttir.empty() : tensor<1x8x640x128xf32>
    %1386 = "ttir.multiply"(%1384, %196, %1385) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1387 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1388 = "ttir.typecast"(%1386, %1387) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1389 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1390 = "ttir.add"(%1374, %1388, %1389) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1391 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %1392 = "ttir.reshape"(%1390, %1391) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %1393 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %1394 = "ttir.broadcast"(%1392, %1393) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %1395 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1396 = "ttir.reshape"(%1394, %1395) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1397 = ttir.empty() : tensor<1x24x128x640xbf16>
    %1398 = "ttir.permute"(%1396, %1397) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %1399 = ttir.empty() : tensor<24x128x640xbf16>
    %1400 = "ttir.reshape"(%1398, %1399) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %1401 = "ttir.dot_general"(%1357, %1400) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %1402 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1403 = "ttir.reshape"(%1401, %1402) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1404 = ttir.empty() : tensor<1x24x640x640xf32>
    %1405 = "ttir.typecast"(%1403, %1404) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1406 = ttir.empty() : tensor<1x24x640x640xf32>
    %1407 = "ttir.multiply"(%1405, %221, %1406) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1408 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1409 = "ttir.typecast"(%1407, %1408) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1410 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1411 = "ttir.add"(%1409, %285, %1410) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1412 = ttir.empty() : tensor<1x24x640x640xf32>
    %1413 = "ttir.typecast"(%1411, %1412) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1414 = ttir.empty() : tensor<1x24x640xf32>
    %1415 = "ttir.max"(%1413, %1414) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %1416 = ttir.empty() : tensor<1x24x640x1xf32>
    %1417 = "ttir.reshape"(%1415, %1416) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %1418 = ttir.empty() : tensor<1x24x640x640xf32>
    %1419 = "ttir.broadcast"(%1417, %1418) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1420 = ttir.empty() : tensor<1x24x640x640xf32>
    %1421 = "ttir.subtract"(%1413, %1419, %1420) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1422 = ttir.empty() : tensor<1x24x640x640xf32>
    %1423 = "ttir.exp"(%1421, %1422) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1424 = ttir.empty() : tensor<1x24x640xf32>
    %1425 = "ttir.sum"(%1423, %1424) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %1426 = ttir.empty() : tensor<1x24x640x1xf32>
    %1427 = "ttir.reshape"(%1425, %1426) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %1428 = ttir.empty() : tensor<1x24x640x640xf32>
    %1429 = "ttir.broadcast"(%1427, %1428) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1430 = ttir.empty() : tensor<1x24x640x640xf32>
    %1431 = "ttir.div"(%1423, %1429, %1430) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1432 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1433 = "ttir.typecast"(%1431, %1432) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1434 = ttir.empty() : tensor<24x640x640xbf16>
    %1435 = "ttir.reshape"(%1433, %1434) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %1436 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %1437 = "ttir.reshape"(%1312, %1436) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %1438 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %1439 = "ttir.broadcast"(%1437, %1438) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %1440 = ttir.empty() : tensor<24x640x128xbf16>
    %1441 = "ttir.reshape"(%1439, %1440) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %1442 = "ttir.dot_general"(%1435, %1441) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %1443 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1444 = "ttir.reshape"(%1442, %1443) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1445 = ttir.empty() : tensor<1x640x24x128xbf16>
    %1446 = "ttir.permute"(%1444, %1445) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %1447 = ttir.empty() : tensor<640x3072xbf16>
    %1448 = "ttir.reshape"(%1446, %1447) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %1449 = ttir.empty() : tensor<1x3072x3072xbf16>
    %1450 = "ttir.reshape"(%arg48, %1449) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %1451 = ttir.empty() : tensor<3072x3072xbf16>
    %1452 = "ttir.reshape"(%1450, %1451) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1453 = ttir.empty() : tensor<3072x3072xbf16>
    %1454 = "ttir.permute"(%1452, %1453) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1455 = "ttir.dot_general"(%1448, %1454) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %1456 = ttir.empty() : tensor<1x640x3072xbf16>
    %1457 = "ttir.reshape"(%1455, %1456) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1458 = ttir.empty() : tensor<1x640x3072xbf16>
    %1459 = "ttir.add"(%1269, %1457, %1458) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1460 = ttir.empty() : tensor<1x1x3072xbf16>
    %1461 = "ttir.reshape"(%arg51, %1460) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %1462 = ttir.empty() : tensor<3072xbf16>
    %1463 = "ttir.reshape"(%1461, %1462) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %1464 = ttir.empty() : tensor<3072xf32>
    %1465 = "ttir.typecast"(%1463, %1464) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %1466 = ttir.empty() : tensor<1x1x3072xf32>
    %1467 = "ttir.reshape"(%1465, %1466) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %1468 = ttir.empty() : tensor<1x640x3072xf32>
    %1469 = "ttir.broadcast"(%1467, %1468) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1470 = ttir.empty() : tensor<1x640x3072xf32>
    %1471 = "ttir.typecast"(%1459, %1470) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1472 = ttir.empty() : tensor<1x640x3072xf32>
    %1473 = "ttir.pow"(%1471, %5, %1472) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1474 = ttir.empty() : tensor<1x640xf32>
    %1475 = "ttir.sum"(%1473, %1474) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1476 = ttir.empty() : tensor<1x640xf32>
    %1477 = "ttir.multiply"(%1475, %4, %1476) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1478 = ttir.empty() : tensor<1x640x1xf32>
    %1479 = "ttir.reshape"(%1477, %1478) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1480 = ttir.empty() : tensor<1x640x1xf32>
    %1481 = "ttir.add"(%1479, %46, %1480) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1482 = ttir.empty() : tensor<1x640x1xf32>
    %1483 = "ttir.rsqrt"(%1481, %1482) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1484 = ttir.empty() : tensor<1x640xf32>
    %1485 = "ttir.reshape"(%1483, %1484) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1486 = ttir.empty() : tensor<1x640x1xf32>
    %1487 = "ttir.reshape"(%1485, %1486) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1488 = ttir.empty() : tensor<1x640x3072xf32>
    %1489 = "ttir.broadcast"(%1487, %1488) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1490 = ttir.empty() : tensor<1x640x3072xf32>
    %1491 = "ttir.multiply"(%1471, %1489, %1490) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1492 = ttir.empty() : tensor<1x640x3072xbf16>
    %1493 = "ttir.typecast"(%1491, %1492) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1494 = ttir.empty() : tensor<1x640x3072xf32>
    %1495 = "ttir.typecast"(%1493, %1494) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1496 = ttir.empty() : tensor<1x640x3072xf32>
    %1497 = "ttir.multiply"(%1469, %1495, %1496) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1498 = ttir.empty() : tensor<1x640x3072xbf16>
    %1499 = "ttir.typecast"(%1497, %1498) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1500 = ttir.empty() : tensor<640x3072xbf16>
    %1501 = "ttir.reshape"(%1499, %1500) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %1502 = ttir.empty() : tensor<1x8192x3072xbf16>
    %1503 = "ttir.reshape"(%arg52, %1502) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %1504 = ttir.empty() : tensor<8192x3072xbf16>
    %1505 = "ttir.reshape"(%1503, %1504) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %1506 = ttir.empty() : tensor<3072x8192xbf16>
    %1507 = "ttir.permute"(%1505, %1506) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %1508 = "ttir.dot_general"(%1501, %1507) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %1509 = ttir.empty() : tensor<1x640x8192xbf16>
    %1510 = "ttir.reshape"(%1508, %1509) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1511 = ttir.empty() : tensor<1x640x8192xf32>
    %1512 = "ttir.typecast"(%1510, %1511) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1513 = ttir.empty() : tensor<1x640x8192xbf16>
    %1514 = "ttir.sigmoid"(%1510, %1513) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1515 = ttir.empty() : tensor<1x640x8192xf32>
    %1516 = "ttir.typecast"(%1514, %1515) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1517 = ttir.empty() : tensor<1x640x8192xf32>
    %1518 = "ttir.multiply"(%1512, %1516, %1517) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1519 = ttir.empty() : tensor<1x640x8192xbf16>
    %1520 = "ttir.typecast"(%1518, %1519) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1521 = ttir.empty() : tensor<1x640x8192xf32>
    %1522 = "ttir.typecast"(%1520, %1521) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1523 = ttir.empty() : tensor<1x8192x3072xbf16>
    %1524 = "ttir.reshape"(%arg47, %1523) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %1525 = ttir.empty() : tensor<8192x3072xbf16>
    %1526 = "ttir.reshape"(%1524, %1525) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %1527 = ttir.empty() : tensor<3072x8192xbf16>
    %1528 = "ttir.permute"(%1526, %1527) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %1529 = "ttir.dot_general"(%1501, %1528) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %1530 = ttir.empty() : tensor<1x640x8192xbf16>
    %1531 = "ttir.reshape"(%1529, %1530) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1532 = ttir.empty() : tensor<1x640x8192xf32>
    %1533 = "ttir.typecast"(%1531, %1532) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1534 = ttir.empty() : tensor<1x640x8192xf32>
    %1535 = "ttir.multiply"(%1522, %1533, %1534) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1536 = ttir.empty() : tensor<1x640x8192xbf16>
    %1537 = "ttir.typecast"(%1535, %1536) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1538 = ttir.empty() : tensor<640x8192xbf16>
    %1539 = "ttir.reshape"(%1537, %1538) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %1540 = ttir.empty() : tensor<1x3072x8192xbf16>
    %1541 = "ttir.reshape"(%arg46, %1540) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %1542 = ttir.empty() : tensor<3072x8192xbf16>
    %1543 = "ttir.reshape"(%1541, %1542) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %1544 = ttir.empty() : tensor<8192x3072xbf16>
    %1545 = "ttir.permute"(%1543, %1544) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %1546 = "ttir.dot_general"(%1539, %1545) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %1547 = ttir.empty() : tensor<1x640x3072xbf16>
    %1548 = "ttir.reshape"(%1546, %1547) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1549 = ttir.empty() : tensor<1x640x3072xbf16>
    %1550 = "ttir.add"(%1459, %1548, %1549) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1551 = ttir.empty() : tensor<1x640x3072xf32>
    %1552 = "ttir.typecast"(%1550, %1551) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1553 = ttir.empty() : tensor<1x640x3072xf32>
    %1554 = "ttir.pow"(%1552, %5, %1553) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1555 = ttir.empty() : tensor<1x640xf32>
    %1556 = "ttir.sum"(%1554, %1555) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1557 = ttir.empty() : tensor<1x640xf32>
    %1558 = "ttir.multiply"(%1556, %4, %1557) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1559 = ttir.empty() : tensor<1x640x1xf32>
    %1560 = "ttir.reshape"(%1558, %1559) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1561 = ttir.empty() : tensor<1x640x1xf32>
    %1562 = "ttir.add"(%1560, %46, %1561) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1563 = ttir.empty() : tensor<1x640x1xf32>
    %1564 = "ttir.rsqrt"(%1562, %1563) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1565 = ttir.empty() : tensor<1x640xf32>
    %1566 = "ttir.reshape"(%1564, %1565) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1567 = ttir.empty() : tensor<1x640x1xf32>
    %1568 = "ttir.reshape"(%1566, %1567) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1569 = ttir.empty() : tensor<1x640x3072xf32>
    %1570 = "ttir.broadcast"(%1568, %1569) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1571 = ttir.empty() : tensor<1x640x3072xf32>
    %1572 = "ttir.multiply"(%1552, %1570, %1571) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1573 = ttir.empty() : tensor<1x640x3072xbf16>
    %1574 = "ttir.typecast"(%1572, %1573) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1575 = ttir.empty() : tensor<1x640x3072xf32>
    %1576 = "ttir.typecast"(%1574, %1575) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1577 = ttir.empty() : tensor<1x640x3072xf32>
    %1578 = "ttir.multiply"(%1322, %1576, %1577) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1579 = ttir.empty() : tensor<1x640x3072xbf16>
    %1580 = "ttir.typecast"(%1578, %1579) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1581 = ttir.empty() : tensor<640x3072xbf16>
    %1582 = "ttir.reshape"(%1580, %1581) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %1583 = ttir.empty() : tensor<1x1024x3072xbf16>
    %1584 = "ttir.reshape"(%arg45, %1583) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %1585 = ttir.empty() : tensor<1024x3072xbf16>
    %1586 = "ttir.reshape"(%1584, %1585) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %1587 = ttir.empty() : tensor<3072x1024xbf16>
    %1588 = "ttir.permute"(%1586, %1587) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %1589 = "ttir.dot_general"(%1582, %1588) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %1590 = ttir.empty() : tensor<1x640x8x128xbf16>
    %1591 = "ttir.reshape"(%1589, %1590) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %1592 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1593 = "ttir.permute"(%1591, %1592) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1594 = ttir.empty() : tensor<1x1x3072xbf16>
    %1595 = "ttir.reshape"(%arg62, %1594) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %1596 = ttir.empty() : tensor<3072xbf16>
    %1597 = "ttir.reshape"(%1595, %1596) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %1598 = ttir.empty() : tensor<3072xf32>
    %1599 = "ttir.typecast"(%1597, %1598) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %1600 = ttir.empty() : tensor<1x1x3072xf32>
    %1601 = "ttir.reshape"(%1599, %1600) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %1602 = ttir.empty() : tensor<1x640x3072xf32>
    %1603 = "ttir.broadcast"(%1601, %1602) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1604 = ttir.empty() : tensor<1x3072x3072xbf16>
    %1605 = "ttir.reshape"(%arg59, %1604) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %1606 = ttir.empty() : tensor<3072x3072xbf16>
    %1607 = "ttir.reshape"(%1605, %1606) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1608 = ttir.empty() : tensor<3072x3072xbf16>
    %1609 = "ttir.permute"(%1607, %1608) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1610 = "ttir.dot_general"(%1582, %1609) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %1611 = ttir.empty() : tensor<1x640x24x128xbf16>
    %1612 = "ttir.reshape"(%1610, %1611) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %1613 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1614 = "ttir.permute"(%1612, %1613) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1615 = ttir.empty() : tensor<1x24x640x128xf32>
    %1616 = "ttir.typecast"(%1614, %1615) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1617 = ttir.empty() : tensor<1x24x640x128xf32>
    %1618 = "ttir.multiply"(%1616, %125, %1617) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1619 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1620 = "ttir.typecast"(%1618, %1619) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1621 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1622 = "ttir.slice_static"(%1614, %1621) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1623 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1624 = "ttir.neg"(%1622, %1623) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1625 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1626 = "ttir.slice_static"(%1614, %1625) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1627 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1628 = "ttir.concat"(%1624, %1626, %1627) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1629 = ttir.empty() : tensor<1x24x640x128xf32>
    %1630 = "ttir.typecast"(%1628, %1629) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1631 = ttir.empty() : tensor<1x24x640x128xf32>
    %1632 = "ttir.multiply"(%1630, %153, %1631) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1633 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1634 = "ttir.typecast"(%1632, %1633) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1635 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1636 = "ttir.add"(%1620, %1634, %1635) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1637 = ttir.empty() : tensor<24x640x128xbf16>
    %1638 = "ttir.reshape"(%1636, %1637) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %1639 = ttir.empty() : tensor<1x1024x3072xbf16>
    %1640 = "ttir.reshape"(%arg58, %1639) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %1641 = ttir.empty() : tensor<1024x3072xbf16>
    %1642 = "ttir.reshape"(%1640, %1641) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %1643 = ttir.empty() : tensor<3072x1024xbf16>
    %1644 = "ttir.permute"(%1642, %1643) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %1645 = "ttir.dot_general"(%1582, %1644) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %1646 = ttir.empty() : tensor<1x640x8x128xbf16>
    %1647 = "ttir.reshape"(%1645, %1646) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %1648 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1649 = "ttir.permute"(%1647, %1648) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1650 = ttir.empty() : tensor<1x8x640x128xf32>
    %1651 = "ttir.typecast"(%1649, %1650) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1652 = ttir.empty() : tensor<1x8x640x128xf32>
    %1653 = "ttir.multiply"(%1651, %178, %1652) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1654 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1655 = "ttir.typecast"(%1653, %1654) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1656 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1657 = "ttir.slice_static"(%1649, %1656) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1658 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1659 = "ttir.neg"(%1657, %1658) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1660 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1661 = "ttir.slice_static"(%1649, %1660) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1662 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1663 = "ttir.concat"(%1659, %1661, %1662) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1664 = ttir.empty() : tensor<1x8x640x128xf32>
    %1665 = "ttir.typecast"(%1663, %1664) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1666 = ttir.empty() : tensor<1x8x640x128xf32>
    %1667 = "ttir.multiply"(%1665, %196, %1666) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1668 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1669 = "ttir.typecast"(%1667, %1668) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1670 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1671 = "ttir.add"(%1655, %1669, %1670) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1672 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %1673 = "ttir.reshape"(%1671, %1672) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %1674 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %1675 = "ttir.broadcast"(%1673, %1674) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %1676 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1677 = "ttir.reshape"(%1675, %1676) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1678 = ttir.empty() : tensor<1x24x128x640xbf16>
    %1679 = "ttir.permute"(%1677, %1678) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %1680 = ttir.empty() : tensor<24x128x640xbf16>
    %1681 = "ttir.reshape"(%1679, %1680) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %1682 = "ttir.dot_general"(%1638, %1681) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %1683 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1684 = "ttir.reshape"(%1682, %1683) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1685 = ttir.empty() : tensor<1x24x640x640xf32>
    %1686 = "ttir.typecast"(%1684, %1685) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1687 = ttir.empty() : tensor<1x24x640x640xf32>
    %1688 = "ttir.multiply"(%1686, %221, %1687) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1689 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1690 = "ttir.typecast"(%1688, %1689) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1691 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1692 = "ttir.add"(%1690, %285, %1691) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1693 = ttir.empty() : tensor<1x24x640x640xf32>
    %1694 = "ttir.typecast"(%1692, %1693) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1695 = ttir.empty() : tensor<1x24x640xf32>
    %1696 = "ttir.max"(%1694, %1695) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %1697 = ttir.empty() : tensor<1x24x640x1xf32>
    %1698 = "ttir.reshape"(%1696, %1697) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %1699 = ttir.empty() : tensor<1x24x640x640xf32>
    %1700 = "ttir.broadcast"(%1698, %1699) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1701 = ttir.empty() : tensor<1x24x640x640xf32>
    %1702 = "ttir.subtract"(%1694, %1700, %1701) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1703 = ttir.empty() : tensor<1x24x640x640xf32>
    %1704 = "ttir.exp"(%1702, %1703) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1705 = ttir.empty() : tensor<1x24x640xf32>
    %1706 = "ttir.sum"(%1704, %1705) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %1707 = ttir.empty() : tensor<1x24x640x1xf32>
    %1708 = "ttir.reshape"(%1706, %1707) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %1709 = ttir.empty() : tensor<1x24x640x640xf32>
    %1710 = "ttir.broadcast"(%1708, %1709) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1711 = ttir.empty() : tensor<1x24x640x640xf32>
    %1712 = "ttir.div"(%1704, %1710, %1711) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1713 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1714 = "ttir.typecast"(%1712, %1713) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1715 = ttir.empty() : tensor<24x640x640xbf16>
    %1716 = "ttir.reshape"(%1714, %1715) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %1717 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %1718 = "ttir.reshape"(%1593, %1717) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %1719 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %1720 = "ttir.broadcast"(%1718, %1719) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %1721 = ttir.empty() : tensor<24x640x128xbf16>
    %1722 = "ttir.reshape"(%1720, %1721) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %1723 = "ttir.dot_general"(%1716, %1722) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %1724 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1725 = "ttir.reshape"(%1723, %1724) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1726 = ttir.empty() : tensor<1x640x24x128xbf16>
    %1727 = "ttir.permute"(%1725, %1726) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %1728 = ttir.empty() : tensor<640x3072xbf16>
    %1729 = "ttir.reshape"(%1727, %1728) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %1730 = ttir.empty() : tensor<1x3072x3072xbf16>
    %1731 = "ttir.reshape"(%arg57, %1730) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %1732 = ttir.empty() : tensor<3072x3072xbf16>
    %1733 = "ttir.reshape"(%1731, %1732) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1734 = ttir.empty() : tensor<3072x3072xbf16>
    %1735 = "ttir.permute"(%1733, %1734) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1736 = "ttir.dot_general"(%1729, %1735) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %1737 = ttir.empty() : tensor<1x640x3072xbf16>
    %1738 = "ttir.reshape"(%1736, %1737) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1739 = ttir.empty() : tensor<1x640x3072xbf16>
    %1740 = "ttir.add"(%1550, %1738, %1739) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1741 = ttir.empty() : tensor<1x1x3072xbf16>
    %1742 = "ttir.reshape"(%arg60, %1741) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %1743 = ttir.empty() : tensor<3072xbf16>
    %1744 = "ttir.reshape"(%1742, %1743) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %1745 = ttir.empty() : tensor<3072xf32>
    %1746 = "ttir.typecast"(%1744, %1745) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %1747 = ttir.empty() : tensor<1x1x3072xf32>
    %1748 = "ttir.reshape"(%1746, %1747) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %1749 = ttir.empty() : tensor<1x640x3072xf32>
    %1750 = "ttir.broadcast"(%1748, %1749) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1751 = ttir.empty() : tensor<1x640x3072xf32>
    %1752 = "ttir.typecast"(%1740, %1751) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1753 = ttir.empty() : tensor<1x640x3072xf32>
    %1754 = "ttir.pow"(%1752, %5, %1753) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1755 = ttir.empty() : tensor<1x640xf32>
    %1756 = "ttir.sum"(%1754, %1755) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1757 = ttir.empty() : tensor<1x640xf32>
    %1758 = "ttir.multiply"(%1756, %4, %1757) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1759 = ttir.empty() : tensor<1x640x1xf32>
    %1760 = "ttir.reshape"(%1758, %1759) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1761 = ttir.empty() : tensor<1x640x1xf32>
    %1762 = "ttir.add"(%1760, %46, %1761) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1763 = ttir.empty() : tensor<1x640x1xf32>
    %1764 = "ttir.rsqrt"(%1762, %1763) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1765 = ttir.empty() : tensor<1x640xf32>
    %1766 = "ttir.reshape"(%1764, %1765) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1767 = ttir.empty() : tensor<1x640x1xf32>
    %1768 = "ttir.reshape"(%1766, %1767) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1769 = ttir.empty() : tensor<1x640x3072xf32>
    %1770 = "ttir.broadcast"(%1768, %1769) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1771 = ttir.empty() : tensor<1x640x3072xf32>
    %1772 = "ttir.multiply"(%1752, %1770, %1771) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1773 = ttir.empty() : tensor<1x640x3072xbf16>
    %1774 = "ttir.typecast"(%1772, %1773) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1775 = ttir.empty() : tensor<1x640x3072xf32>
    %1776 = "ttir.typecast"(%1774, %1775) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1777 = ttir.empty() : tensor<1x640x3072xf32>
    %1778 = "ttir.multiply"(%1750, %1776, %1777) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1779 = ttir.empty() : tensor<1x640x3072xbf16>
    %1780 = "ttir.typecast"(%1778, %1779) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1781 = ttir.empty() : tensor<640x3072xbf16>
    %1782 = "ttir.reshape"(%1780, %1781) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %1783 = ttir.empty() : tensor<1x8192x3072xbf16>
    %1784 = "ttir.reshape"(%arg61, %1783) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %1785 = ttir.empty() : tensor<8192x3072xbf16>
    %1786 = "ttir.reshape"(%1784, %1785) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %1787 = ttir.empty() : tensor<3072x8192xbf16>
    %1788 = "ttir.permute"(%1786, %1787) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %1789 = "ttir.dot_general"(%1782, %1788) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %1790 = ttir.empty() : tensor<1x640x8192xbf16>
    %1791 = "ttir.reshape"(%1789, %1790) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1792 = ttir.empty() : tensor<1x640x8192xf32>
    %1793 = "ttir.typecast"(%1791, %1792) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1794 = ttir.empty() : tensor<1x640x8192xbf16>
    %1795 = "ttir.sigmoid"(%1791, %1794) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1796 = ttir.empty() : tensor<1x640x8192xf32>
    %1797 = "ttir.typecast"(%1795, %1796) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1798 = ttir.empty() : tensor<1x640x8192xf32>
    %1799 = "ttir.multiply"(%1793, %1797, %1798) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1800 = ttir.empty() : tensor<1x640x8192xbf16>
    %1801 = "ttir.typecast"(%1799, %1800) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1802 = ttir.empty() : tensor<1x640x8192xf32>
    %1803 = "ttir.typecast"(%1801, %1802) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1804 = ttir.empty() : tensor<1x8192x3072xbf16>
    %1805 = "ttir.reshape"(%arg56, %1804) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %1806 = ttir.empty() : tensor<8192x3072xbf16>
    %1807 = "ttir.reshape"(%1805, %1806) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %1808 = ttir.empty() : tensor<3072x8192xbf16>
    %1809 = "ttir.permute"(%1807, %1808) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %1810 = "ttir.dot_general"(%1782, %1809) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %1811 = ttir.empty() : tensor<1x640x8192xbf16>
    %1812 = "ttir.reshape"(%1810, %1811) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1813 = ttir.empty() : tensor<1x640x8192xf32>
    %1814 = "ttir.typecast"(%1812, %1813) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1815 = ttir.empty() : tensor<1x640x8192xf32>
    %1816 = "ttir.multiply"(%1803, %1814, %1815) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %1817 = ttir.empty() : tensor<1x640x8192xbf16>
    %1818 = "ttir.typecast"(%1816, %1817) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %1819 = ttir.empty() : tensor<640x8192xbf16>
    %1820 = "ttir.reshape"(%1818, %1819) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %1821 = ttir.empty() : tensor<1x3072x8192xbf16>
    %1822 = "ttir.reshape"(%arg55, %1821) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %1823 = ttir.empty() : tensor<3072x8192xbf16>
    %1824 = "ttir.reshape"(%1822, %1823) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %1825 = ttir.empty() : tensor<8192x3072xbf16>
    %1826 = "ttir.permute"(%1824, %1825) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %1827 = "ttir.dot_general"(%1820, %1826) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %1828 = ttir.empty() : tensor<1x640x3072xbf16>
    %1829 = "ttir.reshape"(%1827, %1828) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1830 = ttir.empty() : tensor<1x640x3072xbf16>
    %1831 = "ttir.add"(%1740, %1829, %1830) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1832 = ttir.empty() : tensor<1x640x3072xf32>
    %1833 = "ttir.typecast"(%1831, %1832) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1834 = ttir.empty() : tensor<1x640x3072xf32>
    %1835 = "ttir.pow"(%1833, %5, %1834) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1836 = ttir.empty() : tensor<1x640xf32>
    %1837 = "ttir.sum"(%1835, %1836) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1838 = ttir.empty() : tensor<1x640xf32>
    %1839 = "ttir.multiply"(%1837, %4, %1838) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1840 = ttir.empty() : tensor<1x640x1xf32>
    %1841 = "ttir.reshape"(%1839, %1840) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1842 = ttir.empty() : tensor<1x640x1xf32>
    %1843 = "ttir.add"(%1841, %46, %1842) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1844 = ttir.empty() : tensor<1x640x1xf32>
    %1845 = "ttir.rsqrt"(%1843, %1844) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1846 = ttir.empty() : tensor<1x640xf32>
    %1847 = "ttir.reshape"(%1845, %1846) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1848 = ttir.empty() : tensor<1x640x1xf32>
    %1849 = "ttir.reshape"(%1847, %1848) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %1850 = ttir.empty() : tensor<1x640x3072xf32>
    %1851 = "ttir.broadcast"(%1849, %1850) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1852 = ttir.empty() : tensor<1x640x3072xf32>
    %1853 = "ttir.multiply"(%1833, %1851, %1852) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1854 = ttir.empty() : tensor<1x640x3072xbf16>
    %1855 = "ttir.typecast"(%1853, %1854) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1856 = ttir.empty() : tensor<1x640x3072xf32>
    %1857 = "ttir.typecast"(%1855, %1856) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1858 = ttir.empty() : tensor<1x640x3072xf32>
    %1859 = "ttir.multiply"(%1603, %1857, %1858) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1860 = ttir.empty() : tensor<1x640x3072xbf16>
    %1861 = "ttir.typecast"(%1859, %1860) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %1862 = ttir.empty() : tensor<640x3072xbf16>
    %1863 = "ttir.reshape"(%1861, %1862) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %1864 = ttir.empty() : tensor<1x1024x3072xbf16>
    %1865 = "ttir.reshape"(%arg54, %1864) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %1866 = ttir.empty() : tensor<1024x3072xbf16>
    %1867 = "ttir.reshape"(%1865, %1866) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %1868 = ttir.empty() : tensor<3072x1024xbf16>
    %1869 = "ttir.permute"(%1867, %1868) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %1870 = "ttir.dot_general"(%1863, %1869) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %1871 = ttir.empty() : tensor<1x640x8x128xbf16>
    %1872 = "ttir.reshape"(%1870, %1871) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %1873 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1874 = "ttir.permute"(%1872, %1873) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1875 = ttir.empty() : tensor<1x1x3072xbf16>
    %1876 = "ttir.reshape"(%arg71, %1875) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %1877 = ttir.empty() : tensor<3072xbf16>
    %1878 = "ttir.reshape"(%1876, %1877) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %1879 = ttir.empty() : tensor<3072xf32>
    %1880 = "ttir.typecast"(%1878, %1879) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %1881 = ttir.empty() : tensor<1x1x3072xf32>
    %1882 = "ttir.reshape"(%1880, %1881) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %1883 = ttir.empty() : tensor<1x640x3072xf32>
    %1884 = "ttir.broadcast"(%1882, %1883) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %1885 = ttir.empty() : tensor<1x3072x3072xbf16>
    %1886 = "ttir.reshape"(%arg68, %1885) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %1887 = ttir.empty() : tensor<3072x3072xbf16>
    %1888 = "ttir.reshape"(%1886, %1887) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1889 = ttir.empty() : tensor<3072x3072xbf16>
    %1890 = "ttir.permute"(%1888, %1889) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %1891 = "ttir.dot_general"(%1863, %1890) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %1892 = ttir.empty() : tensor<1x640x24x128xbf16>
    %1893 = "ttir.reshape"(%1891, %1892) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %1894 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1895 = "ttir.permute"(%1893, %1894) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1896 = ttir.empty() : tensor<1x24x640x128xf32>
    %1897 = "ttir.typecast"(%1895, %1896) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1898 = ttir.empty() : tensor<1x24x640x128xf32>
    %1899 = "ttir.multiply"(%1897, %125, %1898) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1900 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1901 = "ttir.typecast"(%1899, %1900) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1902 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1903 = "ttir.slice_static"(%1895, %1902) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1904 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1905 = "ttir.neg"(%1903, %1904) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1906 = ttir.empty() : tensor<1x24x640x64xbf16>
    %1907 = "ttir.slice_static"(%1895, %1906) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %1908 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1909 = "ttir.concat"(%1905, %1907, %1908) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1910 = ttir.empty() : tensor<1x24x640x128xf32>
    %1911 = "ttir.typecast"(%1909, %1910) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1912 = ttir.empty() : tensor<1x24x640x128xf32>
    %1913 = "ttir.multiply"(%1911, %153, %1912) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %1914 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1915 = "ttir.typecast"(%1913, %1914) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1916 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1917 = "ttir.add"(%1901, %1915, %1916) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1918 = ttir.empty() : tensor<24x640x128xbf16>
    %1919 = "ttir.reshape"(%1917, %1918) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %1920 = ttir.empty() : tensor<1x1024x3072xbf16>
    %1921 = "ttir.reshape"(%arg67, %1920) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %1922 = ttir.empty() : tensor<1024x3072xbf16>
    %1923 = "ttir.reshape"(%1921, %1922) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %1924 = ttir.empty() : tensor<3072x1024xbf16>
    %1925 = "ttir.permute"(%1923, %1924) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %1926 = "ttir.dot_general"(%1863, %1925) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %1927 = ttir.empty() : tensor<1x640x8x128xbf16>
    %1928 = "ttir.reshape"(%1926, %1927) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %1929 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1930 = "ttir.permute"(%1928, %1929) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1931 = ttir.empty() : tensor<1x8x640x128xf32>
    %1932 = "ttir.typecast"(%1930, %1931) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1933 = ttir.empty() : tensor<1x8x640x128xf32>
    %1934 = "ttir.multiply"(%1932, %178, %1933) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1935 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1936 = "ttir.typecast"(%1934, %1935) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1937 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1938 = "ttir.slice_static"(%1930, %1937) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1939 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1940 = "ttir.neg"(%1938, %1939) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1941 = ttir.empty() : tensor<1x8x640x64xbf16>
    %1942 = "ttir.slice_static"(%1930, %1941) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %1943 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1944 = "ttir.concat"(%1940, %1942, %1943) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1945 = ttir.empty() : tensor<1x8x640x128xf32>
    %1946 = "ttir.typecast"(%1944, %1945) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1947 = ttir.empty() : tensor<1x8x640x128xf32>
    %1948 = "ttir.multiply"(%1946, %196, %1947) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %1949 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1950 = "ttir.typecast"(%1948, %1949) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1951 = ttir.empty() : tensor<1x8x640x128xbf16>
    %1952 = "ttir.add"(%1936, %1950, %1951) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %1953 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %1954 = "ttir.reshape"(%1952, %1953) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %1955 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %1956 = "ttir.broadcast"(%1954, %1955) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %1957 = ttir.empty() : tensor<1x24x640x128xbf16>
    %1958 = "ttir.reshape"(%1956, %1957) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %1959 = ttir.empty() : tensor<1x24x128x640xbf16>
    %1960 = "ttir.permute"(%1958, %1959) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %1961 = ttir.empty() : tensor<24x128x640xbf16>
    %1962 = "ttir.reshape"(%1960, %1961) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %1963 = "ttir.dot_general"(%1919, %1962) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %1964 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1965 = "ttir.reshape"(%1963, %1964) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1966 = ttir.empty() : tensor<1x24x640x640xf32>
    %1967 = "ttir.typecast"(%1965, %1966) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1968 = ttir.empty() : tensor<1x24x640x640xf32>
    %1969 = "ttir.multiply"(%1967, %221, %1968) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1970 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1971 = "ttir.typecast"(%1969, %1970) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1972 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1973 = "ttir.add"(%1971, %285, %1972) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1974 = ttir.empty() : tensor<1x24x640x640xf32>
    %1975 = "ttir.typecast"(%1973, %1974) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1976 = ttir.empty() : tensor<1x24x640xf32>
    %1977 = "ttir.max"(%1975, %1976) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %1978 = ttir.empty() : tensor<1x24x640x1xf32>
    %1979 = "ttir.reshape"(%1977, %1978) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %1980 = ttir.empty() : tensor<1x24x640x640xf32>
    %1981 = "ttir.broadcast"(%1979, %1980) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1982 = ttir.empty() : tensor<1x24x640x640xf32>
    %1983 = "ttir.subtract"(%1975, %1981, %1982) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1984 = ttir.empty() : tensor<1x24x640x640xf32>
    %1985 = "ttir.exp"(%1983, %1984) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1986 = ttir.empty() : tensor<1x24x640xf32>
    %1987 = "ttir.sum"(%1985, %1986) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %1988 = ttir.empty() : tensor<1x24x640x1xf32>
    %1989 = "ttir.reshape"(%1987, %1988) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %1990 = ttir.empty() : tensor<1x24x640x640xf32>
    %1991 = "ttir.broadcast"(%1989, %1990) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1992 = ttir.empty() : tensor<1x24x640x640xf32>
    %1993 = "ttir.div"(%1985, %1991, %1992) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %1994 = ttir.empty() : tensor<1x24x640x640xbf16>
    %1995 = "ttir.typecast"(%1993, %1994) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %1996 = ttir.empty() : tensor<24x640x640xbf16>
    %1997 = "ttir.reshape"(%1995, %1996) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %1998 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %1999 = "ttir.reshape"(%1874, %1998) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %2000 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %2001 = "ttir.broadcast"(%1999, %2000) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %2002 = ttir.empty() : tensor<24x640x128xbf16>
    %2003 = "ttir.reshape"(%2001, %2002) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2004 = "ttir.dot_general"(%1997, %2003) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2005 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2006 = "ttir.reshape"(%2004, %2005) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2007 = ttir.empty() : tensor<1x640x24x128xbf16>
    %2008 = "ttir.permute"(%2006, %2007) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %2009 = ttir.empty() : tensor<640x3072xbf16>
    %2010 = "ttir.reshape"(%2008, %2009) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2011 = ttir.empty() : tensor<1x3072x3072xbf16>
    %2012 = "ttir.reshape"(%arg66, %2011) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %2013 = ttir.empty() : tensor<3072x3072xbf16>
    %2014 = "ttir.reshape"(%2012, %2013) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2015 = ttir.empty() : tensor<3072x3072xbf16>
    %2016 = "ttir.permute"(%2014, %2015) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2017 = "ttir.dot_general"(%2010, %2016) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %2018 = ttir.empty() : tensor<1x640x3072xbf16>
    %2019 = "ttir.reshape"(%2017, %2018) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2020 = ttir.empty() : tensor<1x640x3072xbf16>
    %2021 = "ttir.add"(%1831, %2019, %2020) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2022 = ttir.empty() : tensor<1x1x3072xbf16>
    %2023 = "ttir.reshape"(%arg69, %2022) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %2024 = ttir.empty() : tensor<3072xbf16>
    %2025 = "ttir.reshape"(%2023, %2024) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %2026 = ttir.empty() : tensor<3072xf32>
    %2027 = "ttir.typecast"(%2025, %2026) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %2028 = ttir.empty() : tensor<1x1x3072xf32>
    %2029 = "ttir.reshape"(%2027, %2028) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %2030 = ttir.empty() : tensor<1x640x3072xf32>
    %2031 = "ttir.broadcast"(%2029, %2030) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2032 = ttir.empty() : tensor<1x640x3072xf32>
    %2033 = "ttir.typecast"(%2021, %2032) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2034 = ttir.empty() : tensor<1x640x3072xf32>
    %2035 = "ttir.pow"(%2033, %5, %2034) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2036 = ttir.empty() : tensor<1x640xf32>
    %2037 = "ttir.sum"(%2035, %2036) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2038 = ttir.empty() : tensor<1x640xf32>
    %2039 = "ttir.multiply"(%2037, %4, %2038) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2040 = ttir.empty() : tensor<1x640x1xf32>
    %2041 = "ttir.reshape"(%2039, %2040) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2042 = ttir.empty() : tensor<1x640x1xf32>
    %2043 = "ttir.add"(%2041, %46, %2042) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2044 = ttir.empty() : tensor<1x640x1xf32>
    %2045 = "ttir.rsqrt"(%2043, %2044) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2046 = ttir.empty() : tensor<1x640xf32>
    %2047 = "ttir.reshape"(%2045, %2046) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2048 = ttir.empty() : tensor<1x640x1xf32>
    %2049 = "ttir.reshape"(%2047, %2048) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2050 = ttir.empty() : tensor<1x640x3072xf32>
    %2051 = "ttir.broadcast"(%2049, %2050) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2052 = ttir.empty() : tensor<1x640x3072xf32>
    %2053 = "ttir.multiply"(%2033, %2051, %2052) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2054 = ttir.empty() : tensor<1x640x3072xbf16>
    %2055 = "ttir.typecast"(%2053, %2054) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2056 = ttir.empty() : tensor<1x640x3072xf32>
    %2057 = "ttir.typecast"(%2055, %2056) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2058 = ttir.empty() : tensor<1x640x3072xf32>
    %2059 = "ttir.multiply"(%2031, %2057, %2058) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2060 = ttir.empty() : tensor<1x640x3072xbf16>
    %2061 = "ttir.typecast"(%2059, %2060) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2062 = ttir.empty() : tensor<640x3072xbf16>
    %2063 = "ttir.reshape"(%2061, %2062) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2064 = ttir.empty() : tensor<1x8192x3072xbf16>
    %2065 = "ttir.reshape"(%arg70, %2064) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %2066 = ttir.empty() : tensor<8192x3072xbf16>
    %2067 = "ttir.reshape"(%2065, %2066) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2068 = ttir.empty() : tensor<3072x8192xbf16>
    %2069 = "ttir.permute"(%2067, %2068) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2070 = "ttir.dot_general"(%2063, %2069) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %2071 = ttir.empty() : tensor<1x640x8192xbf16>
    %2072 = "ttir.reshape"(%2070, %2071) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2073 = ttir.empty() : tensor<1x640x8192xf32>
    %2074 = "ttir.typecast"(%2072, %2073) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2075 = ttir.empty() : tensor<1x640x8192xbf16>
    %2076 = "ttir.sigmoid"(%2072, %2075) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2077 = ttir.empty() : tensor<1x640x8192xf32>
    %2078 = "ttir.typecast"(%2076, %2077) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2079 = ttir.empty() : tensor<1x640x8192xf32>
    %2080 = "ttir.multiply"(%2074, %2078, %2079) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2081 = ttir.empty() : tensor<1x640x8192xbf16>
    %2082 = "ttir.typecast"(%2080, %2081) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2083 = ttir.empty() : tensor<1x640x8192xf32>
    %2084 = "ttir.typecast"(%2082, %2083) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2085 = ttir.empty() : tensor<1x8192x3072xbf16>
    %2086 = "ttir.reshape"(%arg65, %2085) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %2087 = ttir.empty() : tensor<8192x3072xbf16>
    %2088 = "ttir.reshape"(%2086, %2087) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2089 = ttir.empty() : tensor<3072x8192xbf16>
    %2090 = "ttir.permute"(%2088, %2089) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2091 = "ttir.dot_general"(%2063, %2090) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %2092 = ttir.empty() : tensor<1x640x8192xbf16>
    %2093 = "ttir.reshape"(%2091, %2092) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2094 = ttir.empty() : tensor<1x640x8192xf32>
    %2095 = "ttir.typecast"(%2093, %2094) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2096 = ttir.empty() : tensor<1x640x8192xf32>
    %2097 = "ttir.multiply"(%2084, %2095, %2096) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2098 = ttir.empty() : tensor<1x640x8192xbf16>
    %2099 = "ttir.typecast"(%2097, %2098) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2100 = ttir.empty() : tensor<640x8192xbf16>
    %2101 = "ttir.reshape"(%2099, %2100) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %2102 = ttir.empty() : tensor<1x3072x8192xbf16>
    %2103 = "ttir.reshape"(%arg64, %2102) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %2104 = ttir.empty() : tensor<3072x8192xbf16>
    %2105 = "ttir.reshape"(%2103, %2104) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2106 = ttir.empty() : tensor<8192x3072xbf16>
    %2107 = "ttir.permute"(%2105, %2106) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2108 = "ttir.dot_general"(%2101, %2107) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %2109 = ttir.empty() : tensor<1x640x3072xbf16>
    %2110 = "ttir.reshape"(%2108, %2109) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2111 = ttir.empty() : tensor<1x640x3072xbf16>
    %2112 = "ttir.add"(%2021, %2110, %2111) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2113 = ttir.empty() : tensor<1x640x3072xf32>
    %2114 = "ttir.typecast"(%2112, %2113) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2115 = ttir.empty() : tensor<1x640x3072xf32>
    %2116 = "ttir.pow"(%2114, %5, %2115) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2117 = ttir.empty() : tensor<1x640xf32>
    %2118 = "ttir.sum"(%2116, %2117) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2119 = ttir.empty() : tensor<1x640xf32>
    %2120 = "ttir.multiply"(%2118, %4, %2119) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2121 = ttir.empty() : tensor<1x640x1xf32>
    %2122 = "ttir.reshape"(%2120, %2121) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2123 = ttir.empty() : tensor<1x640x1xf32>
    %2124 = "ttir.add"(%2122, %46, %2123) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2125 = ttir.empty() : tensor<1x640x1xf32>
    %2126 = "ttir.rsqrt"(%2124, %2125) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2127 = ttir.empty() : tensor<1x640xf32>
    %2128 = "ttir.reshape"(%2126, %2127) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2129 = ttir.empty() : tensor<1x640x1xf32>
    %2130 = "ttir.reshape"(%2128, %2129) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2131 = ttir.empty() : tensor<1x640x3072xf32>
    %2132 = "ttir.broadcast"(%2130, %2131) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2133 = ttir.empty() : tensor<1x640x3072xf32>
    %2134 = "ttir.multiply"(%2114, %2132, %2133) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2135 = ttir.empty() : tensor<1x640x3072xbf16>
    %2136 = "ttir.typecast"(%2134, %2135) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2137 = ttir.empty() : tensor<1x640x3072xf32>
    %2138 = "ttir.typecast"(%2136, %2137) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2139 = ttir.empty() : tensor<1x640x3072xf32>
    %2140 = "ttir.multiply"(%1884, %2138, %2139) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2141 = ttir.empty() : tensor<1x640x3072xbf16>
    %2142 = "ttir.typecast"(%2140, %2141) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2143 = ttir.empty() : tensor<640x3072xbf16>
    %2144 = "ttir.reshape"(%2142, %2143) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2145 = ttir.empty() : tensor<1x1024x3072xbf16>
    %2146 = "ttir.reshape"(%arg63, %2145) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %2147 = ttir.empty() : tensor<1024x3072xbf16>
    %2148 = "ttir.reshape"(%2146, %2147) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %2149 = ttir.empty() : tensor<3072x1024xbf16>
    %2150 = "ttir.permute"(%2148, %2149) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %2151 = "ttir.dot_general"(%2144, %2150) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %2152 = ttir.empty() : tensor<1x640x8x128xbf16>
    %2153 = "ttir.reshape"(%2151, %2152) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %2154 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2155 = "ttir.permute"(%2153, %2154) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2156 = ttir.empty() : tensor<1x1x3072xbf16>
    %2157 = "ttir.reshape"(%arg80, %2156) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %2158 = ttir.empty() : tensor<3072xbf16>
    %2159 = "ttir.reshape"(%2157, %2158) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %2160 = ttir.empty() : tensor<3072xf32>
    %2161 = "ttir.typecast"(%2159, %2160) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %2162 = ttir.empty() : tensor<1x1x3072xf32>
    %2163 = "ttir.reshape"(%2161, %2162) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %2164 = ttir.empty() : tensor<1x640x3072xf32>
    %2165 = "ttir.broadcast"(%2163, %2164) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2166 = ttir.empty() : tensor<1x3072x3072xbf16>
    %2167 = "ttir.reshape"(%arg77, %2166) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %2168 = ttir.empty() : tensor<3072x3072xbf16>
    %2169 = "ttir.reshape"(%2167, %2168) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2170 = ttir.empty() : tensor<3072x3072xbf16>
    %2171 = "ttir.permute"(%2169, %2170) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2172 = "ttir.dot_general"(%2144, %2171) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %2173 = ttir.empty() : tensor<1x640x24x128xbf16>
    %2174 = "ttir.reshape"(%2172, %2173) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %2175 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2176 = "ttir.permute"(%2174, %2175) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2177 = ttir.empty() : tensor<1x24x640x128xf32>
    %2178 = "ttir.typecast"(%2176, %2177) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2179 = ttir.empty() : tensor<1x24x640x128xf32>
    %2180 = "ttir.multiply"(%2178, %125, %2179) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2181 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2182 = "ttir.typecast"(%2180, %2181) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2183 = ttir.empty() : tensor<1x24x640x64xbf16>
    %2184 = "ttir.slice_static"(%2176, %2183) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %2185 = ttir.empty() : tensor<1x24x640x64xbf16>
    %2186 = "ttir.neg"(%2184, %2185) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %2187 = ttir.empty() : tensor<1x24x640x64xbf16>
    %2188 = "ttir.slice_static"(%2176, %2187) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %2189 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2190 = "ttir.concat"(%2186, %2188, %2189) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2191 = ttir.empty() : tensor<1x24x640x128xf32>
    %2192 = "ttir.typecast"(%2190, %2191) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2193 = ttir.empty() : tensor<1x24x640x128xf32>
    %2194 = "ttir.multiply"(%2192, %153, %2193) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2195 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2196 = "ttir.typecast"(%2194, %2195) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2197 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2198 = "ttir.add"(%2182, %2196, %2197) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2199 = ttir.empty() : tensor<24x640x128xbf16>
    %2200 = "ttir.reshape"(%2198, %2199) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2201 = ttir.empty() : tensor<1x1024x3072xbf16>
    %2202 = "ttir.reshape"(%arg76, %2201) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %2203 = ttir.empty() : tensor<1024x3072xbf16>
    %2204 = "ttir.reshape"(%2202, %2203) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %2205 = ttir.empty() : tensor<3072x1024xbf16>
    %2206 = "ttir.permute"(%2204, %2205) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %2207 = "ttir.dot_general"(%2144, %2206) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %2208 = ttir.empty() : tensor<1x640x8x128xbf16>
    %2209 = "ttir.reshape"(%2207, %2208) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %2210 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2211 = "ttir.permute"(%2209, %2210) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2212 = ttir.empty() : tensor<1x8x640x128xf32>
    %2213 = "ttir.typecast"(%2211, %2212) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2214 = ttir.empty() : tensor<1x8x640x128xf32>
    %2215 = "ttir.multiply"(%2213, %178, %2214) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2216 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2217 = "ttir.typecast"(%2215, %2216) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2218 = ttir.empty() : tensor<1x8x640x64xbf16>
    %2219 = "ttir.slice_static"(%2211, %2218) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %2220 = ttir.empty() : tensor<1x8x640x64xbf16>
    %2221 = "ttir.neg"(%2219, %2220) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %2222 = ttir.empty() : tensor<1x8x640x64xbf16>
    %2223 = "ttir.slice_static"(%2211, %2222) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %2224 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2225 = "ttir.concat"(%2221, %2223, %2224) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2226 = ttir.empty() : tensor<1x8x640x128xf32>
    %2227 = "ttir.typecast"(%2225, %2226) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2228 = ttir.empty() : tensor<1x8x640x128xf32>
    %2229 = "ttir.multiply"(%2227, %196, %2228) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2230 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2231 = "ttir.typecast"(%2229, %2230) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2232 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2233 = "ttir.add"(%2217, %2231, %2232) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2234 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %2235 = "ttir.reshape"(%2233, %2234) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %2236 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %2237 = "ttir.broadcast"(%2235, %2236) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %2238 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2239 = "ttir.reshape"(%2237, %2238) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2240 = ttir.empty() : tensor<1x24x128x640xbf16>
    %2241 = "ttir.permute"(%2239, %2240) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %2242 = ttir.empty() : tensor<24x128x640xbf16>
    %2243 = "ttir.reshape"(%2241, %2242) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %2244 = "ttir.dot_general"(%2200, %2243) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %2245 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2246 = "ttir.reshape"(%2244, %2245) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2247 = ttir.empty() : tensor<1x24x640x640xf32>
    %2248 = "ttir.typecast"(%2246, %2247) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2249 = ttir.empty() : tensor<1x24x640x640xf32>
    %2250 = "ttir.multiply"(%2248, %221, %2249) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2251 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2252 = "ttir.typecast"(%2250, %2251) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2253 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2254 = "ttir.add"(%2252, %285, %2253) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2255 = ttir.empty() : tensor<1x24x640x640xf32>
    %2256 = "ttir.typecast"(%2254, %2255) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2257 = ttir.empty() : tensor<1x24x640xf32>
    %2258 = "ttir.max"(%2256, %2257) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %2259 = ttir.empty() : tensor<1x24x640x1xf32>
    %2260 = "ttir.reshape"(%2258, %2259) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %2261 = ttir.empty() : tensor<1x24x640x640xf32>
    %2262 = "ttir.broadcast"(%2260, %2261) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2263 = ttir.empty() : tensor<1x24x640x640xf32>
    %2264 = "ttir.subtract"(%2256, %2262, %2263) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2265 = ttir.empty() : tensor<1x24x640x640xf32>
    %2266 = "ttir.exp"(%2264, %2265) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2267 = ttir.empty() : tensor<1x24x640xf32>
    %2268 = "ttir.sum"(%2266, %2267) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %2269 = ttir.empty() : tensor<1x24x640x1xf32>
    %2270 = "ttir.reshape"(%2268, %2269) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %2271 = ttir.empty() : tensor<1x24x640x640xf32>
    %2272 = "ttir.broadcast"(%2270, %2271) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2273 = ttir.empty() : tensor<1x24x640x640xf32>
    %2274 = "ttir.div"(%2266, %2272, %2273) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2275 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2276 = "ttir.typecast"(%2274, %2275) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2277 = ttir.empty() : tensor<24x640x640xbf16>
    %2278 = "ttir.reshape"(%2276, %2277) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %2279 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %2280 = "ttir.reshape"(%2155, %2279) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %2281 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %2282 = "ttir.broadcast"(%2280, %2281) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %2283 = ttir.empty() : tensor<24x640x128xbf16>
    %2284 = "ttir.reshape"(%2282, %2283) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2285 = "ttir.dot_general"(%2278, %2284) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2286 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2287 = "ttir.reshape"(%2285, %2286) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2288 = ttir.empty() : tensor<1x640x24x128xbf16>
    %2289 = "ttir.permute"(%2287, %2288) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %2290 = ttir.empty() : tensor<640x3072xbf16>
    %2291 = "ttir.reshape"(%2289, %2290) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2292 = ttir.empty() : tensor<1x3072x3072xbf16>
    %2293 = "ttir.reshape"(%arg75, %2292) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %2294 = ttir.empty() : tensor<3072x3072xbf16>
    %2295 = "ttir.reshape"(%2293, %2294) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2296 = ttir.empty() : tensor<3072x3072xbf16>
    %2297 = "ttir.permute"(%2295, %2296) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2298 = "ttir.dot_general"(%2291, %2297) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %2299 = ttir.empty() : tensor<1x640x3072xbf16>
    %2300 = "ttir.reshape"(%2298, %2299) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2301 = ttir.empty() : tensor<1x640x3072xbf16>
    %2302 = "ttir.add"(%2112, %2300, %2301) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2303 = ttir.empty() : tensor<1x1x3072xbf16>
    %2304 = "ttir.reshape"(%arg78, %2303) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %2305 = ttir.empty() : tensor<3072xbf16>
    %2306 = "ttir.reshape"(%2304, %2305) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %2307 = ttir.empty() : tensor<3072xf32>
    %2308 = "ttir.typecast"(%2306, %2307) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %2309 = ttir.empty() : tensor<1x1x3072xf32>
    %2310 = "ttir.reshape"(%2308, %2309) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %2311 = ttir.empty() : tensor<1x640x3072xf32>
    %2312 = "ttir.broadcast"(%2310, %2311) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2313 = ttir.empty() : tensor<1x640x3072xf32>
    %2314 = "ttir.typecast"(%2302, %2313) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2315 = ttir.empty() : tensor<1x640x3072xf32>
    %2316 = "ttir.pow"(%2314, %5, %2315) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2317 = ttir.empty() : tensor<1x640xf32>
    %2318 = "ttir.sum"(%2316, %2317) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2319 = ttir.empty() : tensor<1x640xf32>
    %2320 = "ttir.multiply"(%2318, %4, %2319) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2321 = ttir.empty() : tensor<1x640x1xf32>
    %2322 = "ttir.reshape"(%2320, %2321) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2323 = ttir.empty() : tensor<1x640x1xf32>
    %2324 = "ttir.add"(%2322, %46, %2323) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2325 = ttir.empty() : tensor<1x640x1xf32>
    %2326 = "ttir.rsqrt"(%2324, %2325) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2327 = ttir.empty() : tensor<1x640xf32>
    %2328 = "ttir.reshape"(%2326, %2327) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2329 = ttir.empty() : tensor<1x640x1xf32>
    %2330 = "ttir.reshape"(%2328, %2329) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2331 = ttir.empty() : tensor<1x640x3072xf32>
    %2332 = "ttir.broadcast"(%2330, %2331) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2333 = ttir.empty() : tensor<1x640x3072xf32>
    %2334 = "ttir.multiply"(%2314, %2332, %2333) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2335 = ttir.empty() : tensor<1x640x3072xbf16>
    %2336 = "ttir.typecast"(%2334, %2335) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2337 = ttir.empty() : tensor<1x640x3072xf32>
    %2338 = "ttir.typecast"(%2336, %2337) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2339 = ttir.empty() : tensor<1x640x3072xf32>
    %2340 = "ttir.multiply"(%2312, %2338, %2339) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2341 = ttir.empty() : tensor<1x640x3072xbf16>
    %2342 = "ttir.typecast"(%2340, %2341) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2343 = ttir.empty() : tensor<640x3072xbf16>
    %2344 = "ttir.reshape"(%2342, %2343) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2345 = ttir.empty() : tensor<1x8192x3072xbf16>
    %2346 = "ttir.reshape"(%arg79, %2345) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %2347 = ttir.empty() : tensor<8192x3072xbf16>
    %2348 = "ttir.reshape"(%2346, %2347) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2349 = ttir.empty() : tensor<3072x8192xbf16>
    %2350 = "ttir.permute"(%2348, %2349) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2351 = "ttir.dot_general"(%2344, %2350) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %2352 = ttir.empty() : tensor<1x640x8192xbf16>
    %2353 = "ttir.reshape"(%2351, %2352) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2354 = ttir.empty() : tensor<1x640x8192xf32>
    %2355 = "ttir.typecast"(%2353, %2354) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2356 = ttir.empty() : tensor<1x640x8192xbf16>
    %2357 = "ttir.sigmoid"(%2353, %2356) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2358 = ttir.empty() : tensor<1x640x8192xf32>
    %2359 = "ttir.typecast"(%2357, %2358) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2360 = ttir.empty() : tensor<1x640x8192xf32>
    %2361 = "ttir.multiply"(%2355, %2359, %2360) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2362 = ttir.empty() : tensor<1x640x8192xbf16>
    %2363 = "ttir.typecast"(%2361, %2362) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2364 = ttir.empty() : tensor<1x640x8192xf32>
    %2365 = "ttir.typecast"(%2363, %2364) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2366 = ttir.empty() : tensor<1x8192x3072xbf16>
    %2367 = "ttir.reshape"(%arg74, %2366) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %2368 = ttir.empty() : tensor<8192x3072xbf16>
    %2369 = "ttir.reshape"(%2367, %2368) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2370 = ttir.empty() : tensor<3072x8192xbf16>
    %2371 = "ttir.permute"(%2369, %2370) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2372 = "ttir.dot_general"(%2344, %2371) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %2373 = ttir.empty() : tensor<1x640x8192xbf16>
    %2374 = "ttir.reshape"(%2372, %2373) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2375 = ttir.empty() : tensor<1x640x8192xf32>
    %2376 = "ttir.typecast"(%2374, %2375) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2377 = ttir.empty() : tensor<1x640x8192xf32>
    %2378 = "ttir.multiply"(%2365, %2376, %2377) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2379 = ttir.empty() : tensor<1x640x8192xbf16>
    %2380 = "ttir.typecast"(%2378, %2379) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2381 = ttir.empty() : tensor<640x8192xbf16>
    %2382 = "ttir.reshape"(%2380, %2381) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %2383 = ttir.empty() : tensor<1x3072x8192xbf16>
    %2384 = "ttir.reshape"(%arg73, %2383) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %2385 = ttir.empty() : tensor<3072x8192xbf16>
    %2386 = "ttir.reshape"(%2384, %2385) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2387 = ttir.empty() : tensor<8192x3072xbf16>
    %2388 = "ttir.permute"(%2386, %2387) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2389 = "ttir.dot_general"(%2382, %2388) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %2390 = ttir.empty() : tensor<1x640x3072xbf16>
    %2391 = "ttir.reshape"(%2389, %2390) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2392 = ttir.empty() : tensor<1x640x3072xbf16>
    %2393 = "ttir.add"(%2302, %2391, %2392) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2394 = ttir.empty() : tensor<1x640x3072xf32>
    %2395 = "ttir.typecast"(%2393, %2394) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2396 = ttir.empty() : tensor<1x640x3072xf32>
    %2397 = "ttir.pow"(%2395, %5, %2396) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2398 = ttir.empty() : tensor<1x640xf32>
    %2399 = "ttir.sum"(%2397, %2398) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2400 = ttir.empty() : tensor<1x640xf32>
    %2401 = "ttir.multiply"(%2399, %4, %2400) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2402 = ttir.empty() : tensor<1x640x1xf32>
    %2403 = "ttir.reshape"(%2401, %2402) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2404 = ttir.empty() : tensor<1x640x1xf32>
    %2405 = "ttir.add"(%2403, %46, %2404) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2406 = ttir.empty() : tensor<1x640x1xf32>
    %2407 = "ttir.rsqrt"(%2405, %2406) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2408 = ttir.empty() : tensor<1x640xf32>
    %2409 = "ttir.reshape"(%2407, %2408) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2410 = ttir.empty() : tensor<1x640x1xf32>
    %2411 = "ttir.reshape"(%2409, %2410) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2412 = ttir.empty() : tensor<1x640x3072xf32>
    %2413 = "ttir.broadcast"(%2411, %2412) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2414 = ttir.empty() : tensor<1x640x3072xf32>
    %2415 = "ttir.multiply"(%2395, %2413, %2414) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2416 = ttir.empty() : tensor<1x640x3072xbf16>
    %2417 = "ttir.typecast"(%2415, %2416) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2418 = ttir.empty() : tensor<1x640x3072xf32>
    %2419 = "ttir.typecast"(%2417, %2418) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2420 = ttir.empty() : tensor<1x640x3072xf32>
    %2421 = "ttir.multiply"(%2165, %2419, %2420) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2422 = ttir.empty() : tensor<1x640x3072xbf16>
    %2423 = "ttir.typecast"(%2421, %2422) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2424 = ttir.empty() : tensor<640x3072xbf16>
    %2425 = "ttir.reshape"(%2423, %2424) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2426 = ttir.empty() : tensor<1x1024x3072xbf16>
    %2427 = "ttir.reshape"(%arg72, %2426) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %2428 = ttir.empty() : tensor<1024x3072xbf16>
    %2429 = "ttir.reshape"(%2427, %2428) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %2430 = ttir.empty() : tensor<3072x1024xbf16>
    %2431 = "ttir.permute"(%2429, %2430) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %2432 = "ttir.dot_general"(%2425, %2431) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %2433 = ttir.empty() : tensor<1x640x8x128xbf16>
    %2434 = "ttir.reshape"(%2432, %2433) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %2435 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2436 = "ttir.permute"(%2434, %2435) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2437 = ttir.empty() : tensor<1x1x3072xbf16>
    %2438 = "ttir.reshape"(%arg89, %2437) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %2439 = ttir.empty() : tensor<3072xbf16>
    %2440 = "ttir.reshape"(%2438, %2439) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %2441 = ttir.empty() : tensor<3072xf32>
    %2442 = "ttir.typecast"(%2440, %2441) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %2443 = ttir.empty() : tensor<1x1x3072xf32>
    %2444 = "ttir.reshape"(%2442, %2443) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %2445 = ttir.empty() : tensor<1x640x3072xf32>
    %2446 = "ttir.broadcast"(%2444, %2445) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2447 = ttir.empty() : tensor<1x3072x3072xbf16>
    %2448 = "ttir.reshape"(%arg86, %2447) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %2449 = ttir.empty() : tensor<3072x3072xbf16>
    %2450 = "ttir.reshape"(%2448, %2449) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2451 = ttir.empty() : tensor<3072x3072xbf16>
    %2452 = "ttir.permute"(%2450, %2451) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2453 = "ttir.dot_general"(%2425, %2452) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %2454 = ttir.empty() : tensor<1x640x24x128xbf16>
    %2455 = "ttir.reshape"(%2453, %2454) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %2456 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2457 = "ttir.permute"(%2455, %2456) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2458 = ttir.empty() : tensor<1x24x640x128xf32>
    %2459 = "ttir.typecast"(%2457, %2458) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2460 = ttir.empty() : tensor<1x24x640x128xf32>
    %2461 = "ttir.multiply"(%2459, %125, %2460) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2462 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2463 = "ttir.typecast"(%2461, %2462) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2464 = ttir.empty() : tensor<1x24x640x64xbf16>
    %2465 = "ttir.slice_static"(%2457, %2464) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %2466 = ttir.empty() : tensor<1x24x640x64xbf16>
    %2467 = "ttir.neg"(%2465, %2466) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %2468 = ttir.empty() : tensor<1x24x640x64xbf16>
    %2469 = "ttir.slice_static"(%2457, %2468) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %2470 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2471 = "ttir.concat"(%2467, %2469, %2470) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2472 = ttir.empty() : tensor<1x24x640x128xf32>
    %2473 = "ttir.typecast"(%2471, %2472) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2474 = ttir.empty() : tensor<1x24x640x128xf32>
    %2475 = "ttir.multiply"(%2473, %153, %2474) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2476 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2477 = "ttir.typecast"(%2475, %2476) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2478 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2479 = "ttir.add"(%2463, %2477, %2478) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2480 = ttir.empty() : tensor<24x640x128xbf16>
    %2481 = "ttir.reshape"(%2479, %2480) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2482 = ttir.empty() : tensor<1x1024x3072xbf16>
    %2483 = "ttir.reshape"(%arg85, %2482) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %2484 = ttir.empty() : tensor<1024x3072xbf16>
    %2485 = "ttir.reshape"(%2483, %2484) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %2486 = ttir.empty() : tensor<3072x1024xbf16>
    %2487 = "ttir.permute"(%2485, %2486) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %2488 = "ttir.dot_general"(%2425, %2487) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %2489 = ttir.empty() : tensor<1x640x8x128xbf16>
    %2490 = "ttir.reshape"(%2488, %2489) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %2491 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2492 = "ttir.permute"(%2490, %2491) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2493 = ttir.empty() : tensor<1x8x640x128xf32>
    %2494 = "ttir.typecast"(%2492, %2493) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2495 = ttir.empty() : tensor<1x8x640x128xf32>
    %2496 = "ttir.multiply"(%2494, %178, %2495) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2497 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2498 = "ttir.typecast"(%2496, %2497) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2499 = ttir.empty() : tensor<1x8x640x64xbf16>
    %2500 = "ttir.slice_static"(%2492, %2499) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %2501 = ttir.empty() : tensor<1x8x640x64xbf16>
    %2502 = "ttir.neg"(%2500, %2501) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %2503 = ttir.empty() : tensor<1x8x640x64xbf16>
    %2504 = "ttir.slice_static"(%2492, %2503) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %2505 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2506 = "ttir.concat"(%2502, %2504, %2505) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2507 = ttir.empty() : tensor<1x8x640x128xf32>
    %2508 = "ttir.typecast"(%2506, %2507) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2509 = ttir.empty() : tensor<1x8x640x128xf32>
    %2510 = "ttir.multiply"(%2508, %196, %2509) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2511 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2512 = "ttir.typecast"(%2510, %2511) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2513 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2514 = "ttir.add"(%2498, %2512, %2513) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2515 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %2516 = "ttir.reshape"(%2514, %2515) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %2517 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %2518 = "ttir.broadcast"(%2516, %2517) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %2519 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2520 = "ttir.reshape"(%2518, %2519) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2521 = ttir.empty() : tensor<1x24x128x640xbf16>
    %2522 = "ttir.permute"(%2520, %2521) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %2523 = ttir.empty() : tensor<24x128x640xbf16>
    %2524 = "ttir.reshape"(%2522, %2523) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %2525 = "ttir.dot_general"(%2481, %2524) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %2526 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2527 = "ttir.reshape"(%2525, %2526) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2528 = ttir.empty() : tensor<1x24x640x640xf32>
    %2529 = "ttir.typecast"(%2527, %2528) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2530 = ttir.empty() : tensor<1x24x640x640xf32>
    %2531 = "ttir.multiply"(%2529, %221, %2530) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2532 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2533 = "ttir.typecast"(%2531, %2532) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2534 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2535 = "ttir.add"(%2533, %285, %2534) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2536 = ttir.empty() : tensor<1x24x640x640xf32>
    %2537 = "ttir.typecast"(%2535, %2536) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2538 = ttir.empty() : tensor<1x24x640xf32>
    %2539 = "ttir.max"(%2537, %2538) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %2540 = ttir.empty() : tensor<1x24x640x1xf32>
    %2541 = "ttir.reshape"(%2539, %2540) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %2542 = ttir.empty() : tensor<1x24x640x640xf32>
    %2543 = "ttir.broadcast"(%2541, %2542) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2544 = ttir.empty() : tensor<1x24x640x640xf32>
    %2545 = "ttir.subtract"(%2537, %2543, %2544) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2546 = ttir.empty() : tensor<1x24x640x640xf32>
    %2547 = "ttir.exp"(%2545, %2546) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2548 = ttir.empty() : tensor<1x24x640xf32>
    %2549 = "ttir.sum"(%2547, %2548) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %2550 = ttir.empty() : tensor<1x24x640x1xf32>
    %2551 = "ttir.reshape"(%2549, %2550) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %2552 = ttir.empty() : tensor<1x24x640x640xf32>
    %2553 = "ttir.broadcast"(%2551, %2552) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2554 = ttir.empty() : tensor<1x24x640x640xf32>
    %2555 = "ttir.div"(%2547, %2553, %2554) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2556 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2557 = "ttir.typecast"(%2555, %2556) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2558 = ttir.empty() : tensor<24x640x640xbf16>
    %2559 = "ttir.reshape"(%2557, %2558) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %2560 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %2561 = "ttir.reshape"(%2436, %2560) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %2562 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %2563 = "ttir.broadcast"(%2561, %2562) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %2564 = ttir.empty() : tensor<24x640x128xbf16>
    %2565 = "ttir.reshape"(%2563, %2564) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2566 = "ttir.dot_general"(%2559, %2565) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2567 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2568 = "ttir.reshape"(%2566, %2567) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2569 = ttir.empty() : tensor<1x640x24x128xbf16>
    %2570 = "ttir.permute"(%2568, %2569) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %2571 = ttir.empty() : tensor<640x3072xbf16>
    %2572 = "ttir.reshape"(%2570, %2571) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2573 = ttir.empty() : tensor<1x3072x3072xbf16>
    %2574 = "ttir.reshape"(%arg84, %2573) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %2575 = ttir.empty() : tensor<3072x3072xbf16>
    %2576 = "ttir.reshape"(%2574, %2575) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2577 = ttir.empty() : tensor<3072x3072xbf16>
    %2578 = "ttir.permute"(%2576, %2577) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2579 = "ttir.dot_general"(%2572, %2578) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %2580 = ttir.empty() : tensor<1x640x3072xbf16>
    %2581 = "ttir.reshape"(%2579, %2580) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2582 = ttir.empty() : tensor<1x640x3072xbf16>
    %2583 = "ttir.add"(%2393, %2581, %2582) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2584 = ttir.empty() : tensor<1x1x3072xbf16>
    %2585 = "ttir.reshape"(%arg87, %2584) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %2586 = ttir.empty() : tensor<3072xbf16>
    %2587 = "ttir.reshape"(%2585, %2586) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %2588 = ttir.empty() : tensor<3072xf32>
    %2589 = "ttir.typecast"(%2587, %2588) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %2590 = ttir.empty() : tensor<1x1x3072xf32>
    %2591 = "ttir.reshape"(%2589, %2590) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %2592 = ttir.empty() : tensor<1x640x3072xf32>
    %2593 = "ttir.broadcast"(%2591, %2592) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2594 = ttir.empty() : tensor<1x640x3072xf32>
    %2595 = "ttir.typecast"(%2583, %2594) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2596 = ttir.empty() : tensor<1x640x3072xf32>
    %2597 = "ttir.pow"(%2595, %5, %2596) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2598 = ttir.empty() : tensor<1x640xf32>
    %2599 = "ttir.sum"(%2597, %2598) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2600 = ttir.empty() : tensor<1x640xf32>
    %2601 = "ttir.multiply"(%2599, %4, %2600) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2602 = ttir.empty() : tensor<1x640x1xf32>
    %2603 = "ttir.reshape"(%2601, %2602) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2604 = ttir.empty() : tensor<1x640x1xf32>
    %2605 = "ttir.add"(%2603, %46, %2604) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2606 = ttir.empty() : tensor<1x640x1xf32>
    %2607 = "ttir.rsqrt"(%2605, %2606) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2608 = ttir.empty() : tensor<1x640xf32>
    %2609 = "ttir.reshape"(%2607, %2608) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2610 = ttir.empty() : tensor<1x640x1xf32>
    %2611 = "ttir.reshape"(%2609, %2610) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2612 = ttir.empty() : tensor<1x640x3072xf32>
    %2613 = "ttir.broadcast"(%2611, %2612) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2614 = ttir.empty() : tensor<1x640x3072xf32>
    %2615 = "ttir.multiply"(%2595, %2613, %2614) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2616 = ttir.empty() : tensor<1x640x3072xbf16>
    %2617 = "ttir.typecast"(%2615, %2616) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2618 = ttir.empty() : tensor<1x640x3072xf32>
    %2619 = "ttir.typecast"(%2617, %2618) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2620 = ttir.empty() : tensor<1x640x3072xf32>
    %2621 = "ttir.multiply"(%2593, %2619, %2620) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2622 = ttir.empty() : tensor<1x640x3072xbf16>
    %2623 = "ttir.typecast"(%2621, %2622) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2624 = ttir.empty() : tensor<640x3072xbf16>
    %2625 = "ttir.reshape"(%2623, %2624) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2626 = ttir.empty() : tensor<1x8192x3072xbf16>
    %2627 = "ttir.reshape"(%arg88, %2626) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %2628 = ttir.empty() : tensor<8192x3072xbf16>
    %2629 = "ttir.reshape"(%2627, %2628) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2630 = ttir.empty() : tensor<3072x8192xbf16>
    %2631 = "ttir.permute"(%2629, %2630) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2632 = "ttir.dot_general"(%2625, %2631) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %2633 = ttir.empty() : tensor<1x640x8192xbf16>
    %2634 = "ttir.reshape"(%2632, %2633) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2635 = ttir.empty() : tensor<1x640x8192xf32>
    %2636 = "ttir.typecast"(%2634, %2635) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2637 = ttir.empty() : tensor<1x640x8192xbf16>
    %2638 = "ttir.sigmoid"(%2634, %2637) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2639 = ttir.empty() : tensor<1x640x8192xf32>
    %2640 = "ttir.typecast"(%2638, %2639) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2641 = ttir.empty() : tensor<1x640x8192xf32>
    %2642 = "ttir.multiply"(%2636, %2640, %2641) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2643 = ttir.empty() : tensor<1x640x8192xbf16>
    %2644 = "ttir.typecast"(%2642, %2643) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2645 = ttir.empty() : tensor<1x640x8192xf32>
    %2646 = "ttir.typecast"(%2644, %2645) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2647 = ttir.empty() : tensor<1x8192x3072xbf16>
    %2648 = "ttir.reshape"(%arg83, %2647) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %2649 = ttir.empty() : tensor<8192x3072xbf16>
    %2650 = "ttir.reshape"(%2648, %2649) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2651 = ttir.empty() : tensor<3072x8192xbf16>
    %2652 = "ttir.permute"(%2650, %2651) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2653 = "ttir.dot_general"(%2625, %2652) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %2654 = ttir.empty() : tensor<1x640x8192xbf16>
    %2655 = "ttir.reshape"(%2653, %2654) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2656 = ttir.empty() : tensor<1x640x8192xf32>
    %2657 = "ttir.typecast"(%2655, %2656) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2658 = ttir.empty() : tensor<1x640x8192xf32>
    %2659 = "ttir.multiply"(%2646, %2657, %2658) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2660 = ttir.empty() : tensor<1x640x8192xbf16>
    %2661 = "ttir.typecast"(%2659, %2660) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2662 = ttir.empty() : tensor<640x8192xbf16>
    %2663 = "ttir.reshape"(%2661, %2662) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %2664 = ttir.empty() : tensor<1x3072x8192xbf16>
    %2665 = "ttir.reshape"(%arg82, %2664) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %2666 = ttir.empty() : tensor<3072x8192xbf16>
    %2667 = "ttir.reshape"(%2665, %2666) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2668 = ttir.empty() : tensor<8192x3072xbf16>
    %2669 = "ttir.permute"(%2667, %2668) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2670 = "ttir.dot_general"(%2663, %2669) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %2671 = ttir.empty() : tensor<1x640x3072xbf16>
    %2672 = "ttir.reshape"(%2670, %2671) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2673 = ttir.empty() : tensor<1x640x3072xbf16>
    %2674 = "ttir.add"(%2583, %2672, %2673) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2675 = ttir.empty() : tensor<1x640x3072xf32>
    %2676 = "ttir.typecast"(%2674, %2675) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2677 = ttir.empty() : tensor<1x640x3072xf32>
    %2678 = "ttir.pow"(%2676, %5, %2677) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2679 = ttir.empty() : tensor<1x640xf32>
    %2680 = "ttir.sum"(%2678, %2679) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2681 = ttir.empty() : tensor<1x640xf32>
    %2682 = "ttir.multiply"(%2680, %4, %2681) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2683 = ttir.empty() : tensor<1x640x1xf32>
    %2684 = "ttir.reshape"(%2682, %2683) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2685 = ttir.empty() : tensor<1x640x1xf32>
    %2686 = "ttir.add"(%2684, %46, %2685) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2687 = ttir.empty() : tensor<1x640x1xf32>
    %2688 = "ttir.rsqrt"(%2686, %2687) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2689 = ttir.empty() : tensor<1x640xf32>
    %2690 = "ttir.reshape"(%2688, %2689) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2691 = ttir.empty() : tensor<1x640x1xf32>
    %2692 = "ttir.reshape"(%2690, %2691) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2693 = ttir.empty() : tensor<1x640x3072xf32>
    %2694 = "ttir.broadcast"(%2692, %2693) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2695 = ttir.empty() : tensor<1x640x3072xf32>
    %2696 = "ttir.multiply"(%2676, %2694, %2695) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2697 = ttir.empty() : tensor<1x640x3072xbf16>
    %2698 = "ttir.typecast"(%2696, %2697) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2699 = ttir.empty() : tensor<1x640x3072xf32>
    %2700 = "ttir.typecast"(%2698, %2699) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2701 = ttir.empty() : tensor<1x640x3072xf32>
    %2702 = "ttir.multiply"(%2446, %2700, %2701) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2703 = ttir.empty() : tensor<1x640x3072xbf16>
    %2704 = "ttir.typecast"(%2702, %2703) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2705 = ttir.empty() : tensor<640x3072xbf16>
    %2706 = "ttir.reshape"(%2704, %2705) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2707 = ttir.empty() : tensor<1x1024x3072xbf16>
    %2708 = "ttir.reshape"(%arg81, %2707) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %2709 = ttir.empty() : tensor<1024x3072xbf16>
    %2710 = "ttir.reshape"(%2708, %2709) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %2711 = ttir.empty() : tensor<3072x1024xbf16>
    %2712 = "ttir.permute"(%2710, %2711) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %2713 = "ttir.dot_general"(%2706, %2712) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %2714 = ttir.empty() : tensor<1x640x8x128xbf16>
    %2715 = "ttir.reshape"(%2713, %2714) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %2716 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2717 = "ttir.permute"(%2715, %2716) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2718 = ttir.empty() : tensor<1x1x3072xbf16>
    %2719 = "ttir.reshape"(%arg98, %2718) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %2720 = ttir.empty() : tensor<3072xbf16>
    %2721 = "ttir.reshape"(%2719, %2720) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %2722 = ttir.empty() : tensor<3072xf32>
    %2723 = "ttir.typecast"(%2721, %2722) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %2724 = ttir.empty() : tensor<1x1x3072xf32>
    %2725 = "ttir.reshape"(%2723, %2724) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %2726 = ttir.empty() : tensor<1x640x3072xf32>
    %2727 = "ttir.broadcast"(%2725, %2726) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2728 = ttir.empty() : tensor<1x3072x3072xbf16>
    %2729 = "ttir.reshape"(%arg95, %2728) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %2730 = ttir.empty() : tensor<3072x3072xbf16>
    %2731 = "ttir.reshape"(%2729, %2730) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2732 = ttir.empty() : tensor<3072x3072xbf16>
    %2733 = "ttir.permute"(%2731, %2732) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2734 = "ttir.dot_general"(%2706, %2733) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %2735 = ttir.empty() : tensor<1x640x24x128xbf16>
    %2736 = "ttir.reshape"(%2734, %2735) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %2737 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2738 = "ttir.permute"(%2736, %2737) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2739 = ttir.empty() : tensor<1x24x640x128xf32>
    %2740 = "ttir.typecast"(%2738, %2739) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2741 = ttir.empty() : tensor<1x24x640x128xf32>
    %2742 = "ttir.multiply"(%2740, %125, %2741) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2743 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2744 = "ttir.typecast"(%2742, %2743) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2745 = ttir.empty() : tensor<1x24x640x64xbf16>
    %2746 = "ttir.slice_static"(%2738, %2745) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %2747 = ttir.empty() : tensor<1x24x640x64xbf16>
    %2748 = "ttir.neg"(%2746, %2747) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %2749 = ttir.empty() : tensor<1x24x640x64xbf16>
    %2750 = "ttir.slice_static"(%2738, %2749) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %2751 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2752 = "ttir.concat"(%2748, %2750, %2751) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2753 = ttir.empty() : tensor<1x24x640x128xf32>
    %2754 = "ttir.typecast"(%2752, %2753) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2755 = ttir.empty() : tensor<1x24x640x128xf32>
    %2756 = "ttir.multiply"(%2754, %153, %2755) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %2757 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2758 = "ttir.typecast"(%2756, %2757) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2759 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2760 = "ttir.add"(%2744, %2758, %2759) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2761 = ttir.empty() : tensor<24x640x128xbf16>
    %2762 = "ttir.reshape"(%2760, %2761) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2763 = ttir.empty() : tensor<1x1024x3072xbf16>
    %2764 = "ttir.reshape"(%arg94, %2763) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %2765 = ttir.empty() : tensor<1024x3072xbf16>
    %2766 = "ttir.reshape"(%2764, %2765) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %2767 = ttir.empty() : tensor<3072x1024xbf16>
    %2768 = "ttir.permute"(%2766, %2767) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %2769 = "ttir.dot_general"(%2706, %2768) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %2770 = ttir.empty() : tensor<1x640x8x128xbf16>
    %2771 = "ttir.reshape"(%2769, %2770) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %2772 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2773 = "ttir.permute"(%2771, %2772) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2774 = ttir.empty() : tensor<1x8x640x128xf32>
    %2775 = "ttir.typecast"(%2773, %2774) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2776 = ttir.empty() : tensor<1x8x640x128xf32>
    %2777 = "ttir.multiply"(%2775, %178, %2776) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2778 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2779 = "ttir.typecast"(%2777, %2778) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2780 = ttir.empty() : tensor<1x8x640x64xbf16>
    %2781 = "ttir.slice_static"(%2773, %2780) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %2782 = ttir.empty() : tensor<1x8x640x64xbf16>
    %2783 = "ttir.neg"(%2781, %2782) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %2784 = ttir.empty() : tensor<1x8x640x64xbf16>
    %2785 = "ttir.slice_static"(%2773, %2784) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %2786 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2787 = "ttir.concat"(%2783, %2785, %2786) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2788 = ttir.empty() : tensor<1x8x640x128xf32>
    %2789 = "ttir.typecast"(%2787, %2788) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2790 = ttir.empty() : tensor<1x8x640x128xf32>
    %2791 = "ttir.multiply"(%2789, %196, %2790) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %2792 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2793 = "ttir.typecast"(%2791, %2792) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2794 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2795 = "ttir.add"(%2779, %2793, %2794) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2796 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %2797 = "ttir.reshape"(%2795, %2796) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %2798 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %2799 = "ttir.broadcast"(%2797, %2798) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %2800 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2801 = "ttir.reshape"(%2799, %2800) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2802 = ttir.empty() : tensor<1x24x128x640xbf16>
    %2803 = "ttir.permute"(%2801, %2802) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %2804 = ttir.empty() : tensor<24x128x640xbf16>
    %2805 = "ttir.reshape"(%2803, %2804) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %2806 = "ttir.dot_general"(%2762, %2805) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %2807 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2808 = "ttir.reshape"(%2806, %2807) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2809 = ttir.empty() : tensor<1x24x640x640xf32>
    %2810 = "ttir.typecast"(%2808, %2809) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2811 = ttir.empty() : tensor<1x24x640x640xf32>
    %2812 = "ttir.multiply"(%2810, %221, %2811) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2813 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2814 = "ttir.typecast"(%2812, %2813) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2815 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2816 = "ttir.add"(%2814, %285, %2815) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2817 = ttir.empty() : tensor<1x24x640x640xf32>
    %2818 = "ttir.typecast"(%2816, %2817) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2819 = ttir.empty() : tensor<1x24x640xf32>
    %2820 = "ttir.max"(%2818, %2819) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %2821 = ttir.empty() : tensor<1x24x640x1xf32>
    %2822 = "ttir.reshape"(%2820, %2821) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %2823 = ttir.empty() : tensor<1x24x640x640xf32>
    %2824 = "ttir.broadcast"(%2822, %2823) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2825 = ttir.empty() : tensor<1x24x640x640xf32>
    %2826 = "ttir.subtract"(%2818, %2824, %2825) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2827 = ttir.empty() : tensor<1x24x640x640xf32>
    %2828 = "ttir.exp"(%2826, %2827) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2829 = ttir.empty() : tensor<1x24x640xf32>
    %2830 = "ttir.sum"(%2828, %2829) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %2831 = ttir.empty() : tensor<1x24x640x1xf32>
    %2832 = "ttir.reshape"(%2830, %2831) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %2833 = ttir.empty() : tensor<1x24x640x640xf32>
    %2834 = "ttir.broadcast"(%2832, %2833) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2835 = ttir.empty() : tensor<1x24x640x640xf32>
    %2836 = "ttir.div"(%2828, %2834, %2835) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %2837 = ttir.empty() : tensor<1x24x640x640xbf16>
    %2838 = "ttir.typecast"(%2836, %2837) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %2839 = ttir.empty() : tensor<24x640x640xbf16>
    %2840 = "ttir.reshape"(%2838, %2839) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %2841 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %2842 = "ttir.reshape"(%2717, %2841) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %2843 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %2844 = "ttir.broadcast"(%2842, %2843) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %2845 = ttir.empty() : tensor<24x640x128xbf16>
    %2846 = "ttir.reshape"(%2844, %2845) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2847 = "ttir.dot_general"(%2840, %2846) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %2848 = ttir.empty() : tensor<1x24x640x128xbf16>
    %2849 = "ttir.reshape"(%2847, %2848) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %2850 = ttir.empty() : tensor<1x640x24x128xbf16>
    %2851 = "ttir.permute"(%2849, %2850) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %2852 = ttir.empty() : tensor<640x3072xbf16>
    %2853 = "ttir.reshape"(%2851, %2852) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2854 = ttir.empty() : tensor<1x3072x3072xbf16>
    %2855 = "ttir.reshape"(%arg93, %2854) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %2856 = ttir.empty() : tensor<3072x3072xbf16>
    %2857 = "ttir.reshape"(%2855, %2856) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2858 = ttir.empty() : tensor<3072x3072xbf16>
    %2859 = "ttir.permute"(%2857, %2858) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %2860 = "ttir.dot_general"(%2853, %2859) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %2861 = ttir.empty() : tensor<1x640x3072xbf16>
    %2862 = "ttir.reshape"(%2860, %2861) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2863 = ttir.empty() : tensor<1x640x3072xbf16>
    %2864 = "ttir.add"(%2674, %2862, %2863) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2865 = ttir.empty() : tensor<1x1x3072xbf16>
    %2866 = "ttir.reshape"(%arg96, %2865) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %2867 = ttir.empty() : tensor<3072xbf16>
    %2868 = "ttir.reshape"(%2866, %2867) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %2869 = ttir.empty() : tensor<3072xf32>
    %2870 = "ttir.typecast"(%2868, %2869) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %2871 = ttir.empty() : tensor<1x1x3072xf32>
    %2872 = "ttir.reshape"(%2870, %2871) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %2873 = ttir.empty() : tensor<1x640x3072xf32>
    %2874 = "ttir.broadcast"(%2872, %2873) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2875 = ttir.empty() : tensor<1x640x3072xf32>
    %2876 = "ttir.typecast"(%2864, %2875) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2877 = ttir.empty() : tensor<1x640x3072xf32>
    %2878 = "ttir.pow"(%2876, %5, %2877) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2879 = ttir.empty() : tensor<1x640xf32>
    %2880 = "ttir.sum"(%2878, %2879) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2881 = ttir.empty() : tensor<1x640xf32>
    %2882 = "ttir.multiply"(%2880, %4, %2881) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2883 = ttir.empty() : tensor<1x640x1xf32>
    %2884 = "ttir.reshape"(%2882, %2883) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2885 = ttir.empty() : tensor<1x640x1xf32>
    %2886 = "ttir.add"(%2884, %46, %2885) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2887 = ttir.empty() : tensor<1x640x1xf32>
    %2888 = "ttir.rsqrt"(%2886, %2887) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2889 = ttir.empty() : tensor<1x640xf32>
    %2890 = "ttir.reshape"(%2888, %2889) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2891 = ttir.empty() : tensor<1x640x1xf32>
    %2892 = "ttir.reshape"(%2890, %2891) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2893 = ttir.empty() : tensor<1x640x3072xf32>
    %2894 = "ttir.broadcast"(%2892, %2893) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2895 = ttir.empty() : tensor<1x640x3072xf32>
    %2896 = "ttir.multiply"(%2876, %2894, %2895) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2897 = ttir.empty() : tensor<1x640x3072xbf16>
    %2898 = "ttir.typecast"(%2896, %2897) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2899 = ttir.empty() : tensor<1x640x3072xf32>
    %2900 = "ttir.typecast"(%2898, %2899) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2901 = ttir.empty() : tensor<1x640x3072xf32>
    %2902 = "ttir.multiply"(%2874, %2900, %2901) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2903 = ttir.empty() : tensor<1x640x3072xbf16>
    %2904 = "ttir.typecast"(%2902, %2903) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2905 = ttir.empty() : tensor<640x3072xbf16>
    %2906 = "ttir.reshape"(%2904, %2905) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2907 = ttir.empty() : tensor<1x8192x3072xbf16>
    %2908 = "ttir.reshape"(%arg97, %2907) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %2909 = ttir.empty() : tensor<8192x3072xbf16>
    %2910 = "ttir.reshape"(%2908, %2909) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2911 = ttir.empty() : tensor<3072x8192xbf16>
    %2912 = "ttir.permute"(%2910, %2911) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2913 = "ttir.dot_general"(%2906, %2912) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %2914 = ttir.empty() : tensor<1x640x8192xbf16>
    %2915 = "ttir.reshape"(%2913, %2914) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2916 = ttir.empty() : tensor<1x640x8192xf32>
    %2917 = "ttir.typecast"(%2915, %2916) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2918 = ttir.empty() : tensor<1x640x8192xbf16>
    %2919 = "ttir.sigmoid"(%2915, %2918) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2920 = ttir.empty() : tensor<1x640x8192xf32>
    %2921 = "ttir.typecast"(%2919, %2920) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2922 = ttir.empty() : tensor<1x640x8192xf32>
    %2923 = "ttir.multiply"(%2917, %2921, %2922) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2924 = ttir.empty() : tensor<1x640x8192xbf16>
    %2925 = "ttir.typecast"(%2923, %2924) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2926 = ttir.empty() : tensor<1x640x8192xf32>
    %2927 = "ttir.typecast"(%2925, %2926) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2928 = ttir.empty() : tensor<1x8192x3072xbf16>
    %2929 = "ttir.reshape"(%arg92, %2928) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %2930 = ttir.empty() : tensor<8192x3072xbf16>
    %2931 = "ttir.reshape"(%2929, %2930) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2932 = ttir.empty() : tensor<3072x8192xbf16>
    %2933 = "ttir.permute"(%2931, %2932) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2934 = "ttir.dot_general"(%2906, %2933) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %2935 = ttir.empty() : tensor<1x640x8192xbf16>
    %2936 = "ttir.reshape"(%2934, %2935) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2937 = ttir.empty() : tensor<1x640x8192xf32>
    %2938 = "ttir.typecast"(%2936, %2937) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2939 = ttir.empty() : tensor<1x640x8192xf32>
    %2940 = "ttir.multiply"(%2927, %2938, %2939) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %2941 = ttir.empty() : tensor<1x640x8192xbf16>
    %2942 = "ttir.typecast"(%2940, %2941) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %2943 = ttir.empty() : tensor<640x8192xbf16>
    %2944 = "ttir.reshape"(%2942, %2943) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %2945 = ttir.empty() : tensor<1x3072x8192xbf16>
    %2946 = "ttir.reshape"(%arg91, %2945) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %2947 = ttir.empty() : tensor<3072x8192xbf16>
    %2948 = "ttir.reshape"(%2946, %2947) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %2949 = ttir.empty() : tensor<8192x3072xbf16>
    %2950 = "ttir.permute"(%2948, %2949) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %2951 = "ttir.dot_general"(%2944, %2950) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %2952 = ttir.empty() : tensor<1x640x3072xbf16>
    %2953 = "ttir.reshape"(%2951, %2952) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2954 = ttir.empty() : tensor<1x640x3072xbf16>
    %2955 = "ttir.add"(%2864, %2953, %2954) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2956 = ttir.empty() : tensor<1x640x3072xf32>
    %2957 = "ttir.typecast"(%2955, %2956) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2958 = ttir.empty() : tensor<1x640x3072xf32>
    %2959 = "ttir.pow"(%2957, %5, %2958) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2960 = ttir.empty() : tensor<1x640xf32>
    %2961 = "ttir.sum"(%2959, %2960) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2962 = ttir.empty() : tensor<1x640xf32>
    %2963 = "ttir.multiply"(%2961, %4, %2962) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2964 = ttir.empty() : tensor<1x640x1xf32>
    %2965 = "ttir.reshape"(%2963, %2964) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2966 = ttir.empty() : tensor<1x640x1xf32>
    %2967 = "ttir.add"(%2965, %46, %2966) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2968 = ttir.empty() : tensor<1x640x1xf32>
    %2969 = "ttir.rsqrt"(%2967, %2968) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2970 = ttir.empty() : tensor<1x640xf32>
    %2971 = "ttir.reshape"(%2969, %2970) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %2972 = ttir.empty() : tensor<1x640x1xf32>
    %2973 = "ttir.reshape"(%2971, %2972) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %2974 = ttir.empty() : tensor<1x640x3072xf32>
    %2975 = "ttir.broadcast"(%2973, %2974) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2976 = ttir.empty() : tensor<1x640x3072xf32>
    %2977 = "ttir.multiply"(%2957, %2975, %2976) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2978 = ttir.empty() : tensor<1x640x3072xbf16>
    %2979 = "ttir.typecast"(%2977, %2978) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2980 = ttir.empty() : tensor<1x640x3072xf32>
    %2981 = "ttir.typecast"(%2979, %2980) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2982 = ttir.empty() : tensor<1x640x3072xf32>
    %2983 = "ttir.multiply"(%2727, %2981, %2982) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %2984 = ttir.empty() : tensor<1x640x3072xbf16>
    %2985 = "ttir.typecast"(%2983, %2984) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %2986 = ttir.empty() : tensor<640x3072xbf16>
    %2987 = "ttir.reshape"(%2985, %2986) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %2988 = ttir.empty() : tensor<1x1024x3072xbf16>
    %2989 = "ttir.reshape"(%arg90, %2988) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %2990 = ttir.empty() : tensor<1024x3072xbf16>
    %2991 = "ttir.reshape"(%2989, %2990) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %2992 = ttir.empty() : tensor<3072x1024xbf16>
    %2993 = "ttir.permute"(%2991, %2992) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %2994 = "ttir.dot_general"(%2987, %2993) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %2995 = ttir.empty() : tensor<1x640x8x128xbf16>
    %2996 = "ttir.reshape"(%2994, %2995) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %2997 = ttir.empty() : tensor<1x8x640x128xbf16>
    %2998 = "ttir.permute"(%2996, %2997) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %2999 = ttir.empty() : tensor<1x1x3072xbf16>
    %3000 = "ttir.reshape"(%arg107, %2999) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %3001 = ttir.empty() : tensor<3072xbf16>
    %3002 = "ttir.reshape"(%3000, %3001) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %3003 = ttir.empty() : tensor<3072xf32>
    %3004 = "ttir.typecast"(%3002, %3003) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %3005 = ttir.empty() : tensor<1x1x3072xf32>
    %3006 = "ttir.reshape"(%3004, %3005) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %3007 = ttir.empty() : tensor<1x640x3072xf32>
    %3008 = "ttir.broadcast"(%3006, %3007) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3009 = ttir.empty() : tensor<1x3072x3072xbf16>
    %3010 = "ttir.reshape"(%arg104, %3009) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %3011 = ttir.empty() : tensor<3072x3072xbf16>
    %3012 = "ttir.reshape"(%3010, %3011) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3013 = ttir.empty() : tensor<3072x3072xbf16>
    %3014 = "ttir.permute"(%3012, %3013) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3015 = "ttir.dot_general"(%2987, %3014) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %3016 = ttir.empty() : tensor<1x640x24x128xbf16>
    %3017 = "ttir.reshape"(%3015, %3016) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %3018 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3019 = "ttir.permute"(%3017, %3018) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3020 = ttir.empty() : tensor<1x24x640x128xf32>
    %3021 = "ttir.typecast"(%3019, %3020) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3022 = ttir.empty() : tensor<1x24x640x128xf32>
    %3023 = "ttir.multiply"(%3021, %125, %3022) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3024 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3025 = "ttir.typecast"(%3023, %3024) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3026 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3027 = "ttir.slice_static"(%3019, %3026) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3028 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3029 = "ttir.neg"(%3027, %3028) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3030 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3031 = "ttir.slice_static"(%3019, %3030) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3032 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3033 = "ttir.concat"(%3029, %3031, %3032) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3034 = ttir.empty() : tensor<1x24x640x128xf32>
    %3035 = "ttir.typecast"(%3033, %3034) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3036 = ttir.empty() : tensor<1x24x640x128xf32>
    %3037 = "ttir.multiply"(%3035, %153, %3036) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3038 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3039 = "ttir.typecast"(%3037, %3038) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3040 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3041 = "ttir.add"(%3025, %3039, %3040) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3042 = ttir.empty() : tensor<24x640x128xbf16>
    %3043 = "ttir.reshape"(%3041, %3042) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3044 = ttir.empty() : tensor<1x1024x3072xbf16>
    %3045 = "ttir.reshape"(%arg103, %3044) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %3046 = ttir.empty() : tensor<1024x3072xbf16>
    %3047 = "ttir.reshape"(%3045, %3046) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %3048 = ttir.empty() : tensor<3072x1024xbf16>
    %3049 = "ttir.permute"(%3047, %3048) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %3050 = "ttir.dot_general"(%2987, %3049) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %3051 = ttir.empty() : tensor<1x640x8x128xbf16>
    %3052 = "ttir.reshape"(%3050, %3051) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %3053 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3054 = "ttir.permute"(%3052, %3053) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3055 = ttir.empty() : tensor<1x8x640x128xf32>
    %3056 = "ttir.typecast"(%3054, %3055) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3057 = ttir.empty() : tensor<1x8x640x128xf32>
    %3058 = "ttir.multiply"(%3056, %178, %3057) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3059 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3060 = "ttir.typecast"(%3058, %3059) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3061 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3062 = "ttir.slice_static"(%3054, %3061) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3063 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3064 = "ttir.neg"(%3062, %3063) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3065 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3066 = "ttir.slice_static"(%3054, %3065) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3067 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3068 = "ttir.concat"(%3064, %3066, %3067) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3069 = ttir.empty() : tensor<1x8x640x128xf32>
    %3070 = "ttir.typecast"(%3068, %3069) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3071 = ttir.empty() : tensor<1x8x640x128xf32>
    %3072 = "ttir.multiply"(%3070, %196, %3071) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3073 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3074 = "ttir.typecast"(%3072, %3073) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3075 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3076 = "ttir.add"(%3060, %3074, %3075) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3077 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %3078 = "ttir.reshape"(%3076, %3077) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %3079 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %3080 = "ttir.broadcast"(%3078, %3079) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %3081 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3082 = "ttir.reshape"(%3080, %3081) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3083 = ttir.empty() : tensor<1x24x128x640xbf16>
    %3084 = "ttir.permute"(%3082, %3083) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %3085 = ttir.empty() : tensor<24x128x640xbf16>
    %3086 = "ttir.reshape"(%3084, %3085) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %3087 = "ttir.dot_general"(%3043, %3086) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %3088 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3089 = "ttir.reshape"(%3087, %3088) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3090 = ttir.empty() : tensor<1x24x640x640xf32>
    %3091 = "ttir.typecast"(%3089, %3090) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3092 = ttir.empty() : tensor<1x24x640x640xf32>
    %3093 = "ttir.multiply"(%3091, %221, %3092) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3094 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3095 = "ttir.typecast"(%3093, %3094) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3096 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3097 = "ttir.add"(%3095, %285, %3096) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3098 = ttir.empty() : tensor<1x24x640x640xf32>
    %3099 = "ttir.typecast"(%3097, %3098) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3100 = ttir.empty() : tensor<1x24x640xf32>
    %3101 = "ttir.max"(%3099, %3100) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %3102 = ttir.empty() : tensor<1x24x640x1xf32>
    %3103 = "ttir.reshape"(%3101, %3102) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %3104 = ttir.empty() : tensor<1x24x640x640xf32>
    %3105 = "ttir.broadcast"(%3103, %3104) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3106 = ttir.empty() : tensor<1x24x640x640xf32>
    %3107 = "ttir.subtract"(%3099, %3105, %3106) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3108 = ttir.empty() : tensor<1x24x640x640xf32>
    %3109 = "ttir.exp"(%3107, %3108) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3110 = ttir.empty() : tensor<1x24x640xf32>
    %3111 = "ttir.sum"(%3109, %3110) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %3112 = ttir.empty() : tensor<1x24x640x1xf32>
    %3113 = "ttir.reshape"(%3111, %3112) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %3114 = ttir.empty() : tensor<1x24x640x640xf32>
    %3115 = "ttir.broadcast"(%3113, %3114) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3116 = ttir.empty() : tensor<1x24x640x640xf32>
    %3117 = "ttir.div"(%3109, %3115, %3116) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3118 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3119 = "ttir.typecast"(%3117, %3118) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3120 = ttir.empty() : tensor<24x640x640xbf16>
    %3121 = "ttir.reshape"(%3119, %3120) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %3122 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %3123 = "ttir.reshape"(%2998, %3122) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %3124 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %3125 = "ttir.broadcast"(%3123, %3124) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %3126 = ttir.empty() : tensor<24x640x128xbf16>
    %3127 = "ttir.reshape"(%3125, %3126) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3128 = "ttir.dot_general"(%3121, %3127) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3129 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3130 = "ttir.reshape"(%3128, %3129) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3131 = ttir.empty() : tensor<1x640x24x128xbf16>
    %3132 = "ttir.permute"(%3130, %3131) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %3133 = ttir.empty() : tensor<640x3072xbf16>
    %3134 = "ttir.reshape"(%3132, %3133) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %3135 = ttir.empty() : tensor<1x3072x3072xbf16>
    %3136 = "ttir.reshape"(%arg102, %3135) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %3137 = ttir.empty() : tensor<3072x3072xbf16>
    %3138 = "ttir.reshape"(%3136, %3137) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3139 = ttir.empty() : tensor<3072x3072xbf16>
    %3140 = "ttir.permute"(%3138, %3139) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3141 = "ttir.dot_general"(%3134, %3140) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %3142 = ttir.empty() : tensor<1x640x3072xbf16>
    %3143 = "ttir.reshape"(%3141, %3142) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3144 = ttir.empty() : tensor<1x640x3072xbf16>
    %3145 = "ttir.add"(%2955, %3143, %3144) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3146 = ttir.empty() : tensor<1x1x3072xbf16>
    %3147 = "ttir.reshape"(%arg105, %3146) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %3148 = ttir.empty() : tensor<3072xbf16>
    %3149 = "ttir.reshape"(%3147, %3148) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %3150 = ttir.empty() : tensor<3072xf32>
    %3151 = "ttir.typecast"(%3149, %3150) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %3152 = ttir.empty() : tensor<1x1x3072xf32>
    %3153 = "ttir.reshape"(%3151, %3152) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %3154 = ttir.empty() : tensor<1x640x3072xf32>
    %3155 = "ttir.broadcast"(%3153, %3154) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3156 = ttir.empty() : tensor<1x640x3072xf32>
    %3157 = "ttir.typecast"(%3145, %3156) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3158 = ttir.empty() : tensor<1x640x3072xf32>
    %3159 = "ttir.pow"(%3157, %5, %3158) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3160 = ttir.empty() : tensor<1x640xf32>
    %3161 = "ttir.sum"(%3159, %3160) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3162 = ttir.empty() : tensor<1x640xf32>
    %3163 = "ttir.multiply"(%3161, %4, %3162) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3164 = ttir.empty() : tensor<1x640x1xf32>
    %3165 = "ttir.reshape"(%3163, %3164) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3166 = ttir.empty() : tensor<1x640x1xf32>
    %3167 = "ttir.add"(%3165, %46, %3166) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3168 = ttir.empty() : tensor<1x640x1xf32>
    %3169 = "ttir.rsqrt"(%3167, %3168) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3170 = ttir.empty() : tensor<1x640xf32>
    %3171 = "ttir.reshape"(%3169, %3170) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3172 = ttir.empty() : tensor<1x640x1xf32>
    %3173 = "ttir.reshape"(%3171, %3172) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3174 = ttir.empty() : tensor<1x640x3072xf32>
    %3175 = "ttir.broadcast"(%3173, %3174) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3176 = ttir.empty() : tensor<1x640x3072xf32>
    %3177 = "ttir.multiply"(%3157, %3175, %3176) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3178 = ttir.empty() : tensor<1x640x3072xbf16>
    %3179 = "ttir.typecast"(%3177, %3178) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3180 = ttir.empty() : tensor<1x640x3072xf32>
    %3181 = "ttir.typecast"(%3179, %3180) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3182 = ttir.empty() : tensor<1x640x3072xf32>
    %3183 = "ttir.multiply"(%3155, %3181, %3182) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3184 = ttir.empty() : tensor<1x640x3072xbf16>
    %3185 = "ttir.typecast"(%3183, %3184) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3186 = ttir.empty() : tensor<640x3072xbf16>
    %3187 = "ttir.reshape"(%3185, %3186) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %3188 = ttir.empty() : tensor<1x8192x3072xbf16>
    %3189 = "ttir.reshape"(%arg106, %3188) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %3190 = ttir.empty() : tensor<8192x3072xbf16>
    %3191 = "ttir.reshape"(%3189, %3190) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %3192 = ttir.empty() : tensor<3072x8192xbf16>
    %3193 = "ttir.permute"(%3191, %3192) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %3194 = "ttir.dot_general"(%3187, %3193) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %3195 = ttir.empty() : tensor<1x640x8192xbf16>
    %3196 = "ttir.reshape"(%3194, %3195) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3197 = ttir.empty() : tensor<1x640x8192xf32>
    %3198 = "ttir.typecast"(%3196, %3197) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3199 = ttir.empty() : tensor<1x640x8192xbf16>
    %3200 = "ttir.sigmoid"(%3196, %3199) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3201 = ttir.empty() : tensor<1x640x8192xf32>
    %3202 = "ttir.typecast"(%3200, %3201) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3203 = ttir.empty() : tensor<1x640x8192xf32>
    %3204 = "ttir.multiply"(%3198, %3202, %3203) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3205 = ttir.empty() : tensor<1x640x8192xbf16>
    %3206 = "ttir.typecast"(%3204, %3205) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3207 = ttir.empty() : tensor<1x640x8192xf32>
    %3208 = "ttir.typecast"(%3206, %3207) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3209 = ttir.empty() : tensor<1x8192x3072xbf16>
    %3210 = "ttir.reshape"(%arg101, %3209) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %3211 = ttir.empty() : tensor<8192x3072xbf16>
    %3212 = "ttir.reshape"(%3210, %3211) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %3213 = ttir.empty() : tensor<3072x8192xbf16>
    %3214 = "ttir.permute"(%3212, %3213) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %3215 = "ttir.dot_general"(%3187, %3214) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %3216 = ttir.empty() : tensor<1x640x8192xbf16>
    %3217 = "ttir.reshape"(%3215, %3216) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3218 = ttir.empty() : tensor<1x640x8192xf32>
    %3219 = "ttir.typecast"(%3217, %3218) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3220 = ttir.empty() : tensor<1x640x8192xf32>
    %3221 = "ttir.multiply"(%3208, %3219, %3220) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3222 = ttir.empty() : tensor<1x640x8192xbf16>
    %3223 = "ttir.typecast"(%3221, %3222) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3224 = ttir.empty() : tensor<640x8192xbf16>
    %3225 = "ttir.reshape"(%3223, %3224) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %3226 = ttir.empty() : tensor<1x3072x8192xbf16>
    %3227 = "ttir.reshape"(%arg100, %3226) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %3228 = ttir.empty() : tensor<3072x8192xbf16>
    %3229 = "ttir.reshape"(%3227, %3228) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %3230 = ttir.empty() : tensor<8192x3072xbf16>
    %3231 = "ttir.permute"(%3229, %3230) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %3232 = "ttir.dot_general"(%3225, %3231) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %3233 = ttir.empty() : tensor<1x640x3072xbf16>
    %3234 = "ttir.reshape"(%3232, %3233) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3235 = ttir.empty() : tensor<1x640x3072xbf16>
    %3236 = "ttir.add"(%3145, %3234, %3235) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3237 = ttir.empty() : tensor<1x640x3072xf32>
    %3238 = "ttir.typecast"(%3236, %3237) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3239 = ttir.empty() : tensor<1x640x3072xf32>
    %3240 = "ttir.pow"(%3238, %5, %3239) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3241 = ttir.empty() : tensor<1x640xf32>
    %3242 = "ttir.sum"(%3240, %3241) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3243 = ttir.empty() : tensor<1x640xf32>
    %3244 = "ttir.multiply"(%3242, %4, %3243) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3245 = ttir.empty() : tensor<1x640x1xf32>
    %3246 = "ttir.reshape"(%3244, %3245) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3247 = ttir.empty() : tensor<1x640x1xf32>
    %3248 = "ttir.add"(%3246, %46, %3247) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3249 = ttir.empty() : tensor<1x640x1xf32>
    %3250 = "ttir.rsqrt"(%3248, %3249) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3251 = ttir.empty() : tensor<1x640xf32>
    %3252 = "ttir.reshape"(%3250, %3251) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3253 = ttir.empty() : tensor<1x640x1xf32>
    %3254 = "ttir.reshape"(%3252, %3253) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3255 = ttir.empty() : tensor<1x640x3072xf32>
    %3256 = "ttir.broadcast"(%3254, %3255) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3257 = ttir.empty() : tensor<1x640x3072xf32>
    %3258 = "ttir.multiply"(%3238, %3256, %3257) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3259 = ttir.empty() : tensor<1x640x3072xbf16>
    %3260 = "ttir.typecast"(%3258, %3259) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3261 = ttir.empty() : tensor<1x640x3072xf32>
    %3262 = "ttir.typecast"(%3260, %3261) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3263 = ttir.empty() : tensor<1x640x3072xf32>
    %3264 = "ttir.multiply"(%3008, %3262, %3263) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3265 = ttir.empty() : tensor<1x640x3072xbf16>
    %3266 = "ttir.typecast"(%3264, %3265) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3267 = ttir.empty() : tensor<640x3072xbf16>
    %3268 = "ttir.reshape"(%3266, %3267) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %3269 = ttir.empty() : tensor<1x1024x3072xbf16>
    %3270 = "ttir.reshape"(%arg99, %3269) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %3271 = ttir.empty() : tensor<1024x3072xbf16>
    %3272 = "ttir.reshape"(%3270, %3271) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %3273 = ttir.empty() : tensor<3072x1024xbf16>
    %3274 = "ttir.permute"(%3272, %3273) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %3275 = "ttir.dot_general"(%3268, %3274) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %3276 = ttir.empty() : tensor<1x640x8x128xbf16>
    %3277 = "ttir.reshape"(%3275, %3276) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %3278 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3279 = "ttir.permute"(%3277, %3278) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3280 = ttir.empty() : tensor<1x1x3072xbf16>
    %3281 = "ttir.reshape"(%arg116, %3280) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %3282 = ttir.empty() : tensor<3072xbf16>
    %3283 = "ttir.reshape"(%3281, %3282) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %3284 = ttir.empty() : tensor<3072xf32>
    %3285 = "ttir.typecast"(%3283, %3284) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %3286 = ttir.empty() : tensor<1x1x3072xf32>
    %3287 = "ttir.reshape"(%3285, %3286) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %3288 = ttir.empty() : tensor<1x640x3072xf32>
    %3289 = "ttir.broadcast"(%3287, %3288) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3290 = ttir.empty() : tensor<1x3072x3072xbf16>
    %3291 = "ttir.reshape"(%arg113, %3290) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %3292 = ttir.empty() : tensor<3072x3072xbf16>
    %3293 = "ttir.reshape"(%3291, %3292) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3294 = ttir.empty() : tensor<3072x3072xbf16>
    %3295 = "ttir.permute"(%3293, %3294) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3296 = "ttir.dot_general"(%3268, %3295) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %3297 = ttir.empty() : tensor<1x640x24x128xbf16>
    %3298 = "ttir.reshape"(%3296, %3297) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %3299 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3300 = "ttir.permute"(%3298, %3299) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3301 = ttir.empty() : tensor<1x24x640x128xf32>
    %3302 = "ttir.typecast"(%3300, %3301) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3303 = ttir.empty() : tensor<1x24x640x128xf32>
    %3304 = "ttir.multiply"(%3302, %125, %3303) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3305 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3306 = "ttir.typecast"(%3304, %3305) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3307 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3308 = "ttir.slice_static"(%3300, %3307) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3309 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3310 = "ttir.neg"(%3308, %3309) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3311 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3312 = "ttir.slice_static"(%3300, %3311) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3313 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3314 = "ttir.concat"(%3310, %3312, %3313) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3315 = ttir.empty() : tensor<1x24x640x128xf32>
    %3316 = "ttir.typecast"(%3314, %3315) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3317 = ttir.empty() : tensor<1x24x640x128xf32>
    %3318 = "ttir.multiply"(%3316, %153, %3317) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3319 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3320 = "ttir.typecast"(%3318, %3319) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3321 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3322 = "ttir.add"(%3306, %3320, %3321) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3323 = ttir.empty() : tensor<24x640x128xbf16>
    %3324 = "ttir.reshape"(%3322, %3323) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3325 = ttir.empty() : tensor<1x1024x3072xbf16>
    %3326 = "ttir.reshape"(%arg112, %3325) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %3327 = ttir.empty() : tensor<1024x3072xbf16>
    %3328 = "ttir.reshape"(%3326, %3327) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %3329 = ttir.empty() : tensor<3072x1024xbf16>
    %3330 = "ttir.permute"(%3328, %3329) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %3331 = "ttir.dot_general"(%3268, %3330) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %3332 = ttir.empty() : tensor<1x640x8x128xbf16>
    %3333 = "ttir.reshape"(%3331, %3332) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %3334 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3335 = "ttir.permute"(%3333, %3334) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3336 = ttir.empty() : tensor<1x8x640x128xf32>
    %3337 = "ttir.typecast"(%3335, %3336) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3338 = ttir.empty() : tensor<1x8x640x128xf32>
    %3339 = "ttir.multiply"(%3337, %178, %3338) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3340 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3341 = "ttir.typecast"(%3339, %3340) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3342 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3343 = "ttir.slice_static"(%3335, %3342) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3344 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3345 = "ttir.neg"(%3343, %3344) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3346 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3347 = "ttir.slice_static"(%3335, %3346) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3348 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3349 = "ttir.concat"(%3345, %3347, %3348) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3350 = ttir.empty() : tensor<1x8x640x128xf32>
    %3351 = "ttir.typecast"(%3349, %3350) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3352 = ttir.empty() : tensor<1x8x640x128xf32>
    %3353 = "ttir.multiply"(%3351, %196, %3352) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3354 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3355 = "ttir.typecast"(%3353, %3354) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3356 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3357 = "ttir.add"(%3341, %3355, %3356) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3358 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %3359 = "ttir.reshape"(%3357, %3358) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %3360 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %3361 = "ttir.broadcast"(%3359, %3360) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %3362 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3363 = "ttir.reshape"(%3361, %3362) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3364 = ttir.empty() : tensor<1x24x128x640xbf16>
    %3365 = "ttir.permute"(%3363, %3364) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %3366 = ttir.empty() : tensor<24x128x640xbf16>
    %3367 = "ttir.reshape"(%3365, %3366) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %3368 = "ttir.dot_general"(%3324, %3367) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %3369 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3370 = "ttir.reshape"(%3368, %3369) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3371 = ttir.empty() : tensor<1x24x640x640xf32>
    %3372 = "ttir.typecast"(%3370, %3371) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3373 = ttir.empty() : tensor<1x24x640x640xf32>
    %3374 = "ttir.multiply"(%3372, %221, %3373) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3375 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3376 = "ttir.typecast"(%3374, %3375) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3377 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3378 = "ttir.add"(%3376, %285, %3377) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3379 = ttir.empty() : tensor<1x24x640x640xf32>
    %3380 = "ttir.typecast"(%3378, %3379) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3381 = ttir.empty() : tensor<1x24x640xf32>
    %3382 = "ttir.max"(%3380, %3381) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %3383 = ttir.empty() : tensor<1x24x640x1xf32>
    %3384 = "ttir.reshape"(%3382, %3383) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %3385 = ttir.empty() : tensor<1x24x640x640xf32>
    %3386 = "ttir.broadcast"(%3384, %3385) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3387 = ttir.empty() : tensor<1x24x640x640xf32>
    %3388 = "ttir.subtract"(%3380, %3386, %3387) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3389 = ttir.empty() : tensor<1x24x640x640xf32>
    %3390 = "ttir.exp"(%3388, %3389) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3391 = ttir.empty() : tensor<1x24x640xf32>
    %3392 = "ttir.sum"(%3390, %3391) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %3393 = ttir.empty() : tensor<1x24x640x1xf32>
    %3394 = "ttir.reshape"(%3392, %3393) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %3395 = ttir.empty() : tensor<1x24x640x640xf32>
    %3396 = "ttir.broadcast"(%3394, %3395) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3397 = ttir.empty() : tensor<1x24x640x640xf32>
    %3398 = "ttir.div"(%3390, %3396, %3397) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3399 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3400 = "ttir.typecast"(%3398, %3399) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3401 = ttir.empty() : tensor<24x640x640xbf16>
    %3402 = "ttir.reshape"(%3400, %3401) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %3403 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %3404 = "ttir.reshape"(%3279, %3403) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %3405 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %3406 = "ttir.broadcast"(%3404, %3405) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %3407 = ttir.empty() : tensor<24x640x128xbf16>
    %3408 = "ttir.reshape"(%3406, %3407) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3409 = "ttir.dot_general"(%3402, %3408) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3410 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3411 = "ttir.reshape"(%3409, %3410) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3412 = ttir.empty() : tensor<1x640x24x128xbf16>
    %3413 = "ttir.permute"(%3411, %3412) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %3414 = ttir.empty() : tensor<640x3072xbf16>
    %3415 = "ttir.reshape"(%3413, %3414) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %3416 = ttir.empty() : tensor<1x3072x3072xbf16>
    %3417 = "ttir.reshape"(%arg111, %3416) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %3418 = ttir.empty() : tensor<3072x3072xbf16>
    %3419 = "ttir.reshape"(%3417, %3418) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3420 = ttir.empty() : tensor<3072x3072xbf16>
    %3421 = "ttir.permute"(%3419, %3420) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3422 = "ttir.dot_general"(%3415, %3421) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %3423 = ttir.empty() : tensor<1x640x3072xbf16>
    %3424 = "ttir.reshape"(%3422, %3423) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3425 = ttir.empty() : tensor<1x640x3072xbf16>
    %3426 = "ttir.add"(%3236, %3424, %3425) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3427 = ttir.empty() : tensor<1x1x3072xbf16>
    %3428 = "ttir.reshape"(%arg114, %3427) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %3429 = ttir.empty() : tensor<3072xbf16>
    %3430 = "ttir.reshape"(%3428, %3429) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %3431 = ttir.empty() : tensor<3072xf32>
    %3432 = "ttir.typecast"(%3430, %3431) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %3433 = ttir.empty() : tensor<1x1x3072xf32>
    %3434 = "ttir.reshape"(%3432, %3433) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %3435 = ttir.empty() : tensor<1x640x3072xf32>
    %3436 = "ttir.broadcast"(%3434, %3435) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3437 = ttir.empty() : tensor<1x640x3072xf32>
    %3438 = "ttir.typecast"(%3426, %3437) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3439 = ttir.empty() : tensor<1x640x3072xf32>
    %3440 = "ttir.pow"(%3438, %5, %3439) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3441 = ttir.empty() : tensor<1x640xf32>
    %3442 = "ttir.sum"(%3440, %3441) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3443 = ttir.empty() : tensor<1x640xf32>
    %3444 = "ttir.multiply"(%3442, %4, %3443) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3445 = ttir.empty() : tensor<1x640x1xf32>
    %3446 = "ttir.reshape"(%3444, %3445) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3447 = ttir.empty() : tensor<1x640x1xf32>
    %3448 = "ttir.add"(%3446, %46, %3447) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3449 = ttir.empty() : tensor<1x640x1xf32>
    %3450 = "ttir.rsqrt"(%3448, %3449) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3451 = ttir.empty() : tensor<1x640xf32>
    %3452 = "ttir.reshape"(%3450, %3451) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3453 = ttir.empty() : tensor<1x640x1xf32>
    %3454 = "ttir.reshape"(%3452, %3453) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3455 = ttir.empty() : tensor<1x640x3072xf32>
    %3456 = "ttir.broadcast"(%3454, %3455) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3457 = ttir.empty() : tensor<1x640x3072xf32>
    %3458 = "ttir.multiply"(%3438, %3456, %3457) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3459 = ttir.empty() : tensor<1x640x3072xbf16>
    %3460 = "ttir.typecast"(%3458, %3459) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3461 = ttir.empty() : tensor<1x640x3072xf32>
    %3462 = "ttir.typecast"(%3460, %3461) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3463 = ttir.empty() : tensor<1x640x3072xf32>
    %3464 = "ttir.multiply"(%3436, %3462, %3463) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3465 = ttir.empty() : tensor<1x640x3072xbf16>
    %3466 = "ttir.typecast"(%3464, %3465) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3467 = ttir.empty() : tensor<640x3072xbf16>
    %3468 = "ttir.reshape"(%3466, %3467) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %3469 = ttir.empty() : tensor<1x8192x3072xbf16>
    %3470 = "ttir.reshape"(%arg115, %3469) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %3471 = ttir.empty() : tensor<8192x3072xbf16>
    %3472 = "ttir.reshape"(%3470, %3471) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %3473 = ttir.empty() : tensor<3072x8192xbf16>
    %3474 = "ttir.permute"(%3472, %3473) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %3475 = "ttir.dot_general"(%3468, %3474) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %3476 = ttir.empty() : tensor<1x640x8192xbf16>
    %3477 = "ttir.reshape"(%3475, %3476) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3478 = ttir.empty() : tensor<1x640x8192xf32>
    %3479 = "ttir.typecast"(%3477, %3478) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3480 = ttir.empty() : tensor<1x640x8192xbf16>
    %3481 = "ttir.sigmoid"(%3477, %3480) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3482 = ttir.empty() : tensor<1x640x8192xf32>
    %3483 = "ttir.typecast"(%3481, %3482) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3484 = ttir.empty() : tensor<1x640x8192xf32>
    %3485 = "ttir.multiply"(%3479, %3483, %3484) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3486 = ttir.empty() : tensor<1x640x8192xbf16>
    %3487 = "ttir.typecast"(%3485, %3486) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3488 = ttir.empty() : tensor<1x640x8192xf32>
    %3489 = "ttir.typecast"(%3487, %3488) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3490 = ttir.empty() : tensor<1x8192x3072xbf16>
    %3491 = "ttir.reshape"(%arg110, %3490) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %3492 = ttir.empty() : tensor<8192x3072xbf16>
    %3493 = "ttir.reshape"(%3491, %3492) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %3494 = ttir.empty() : tensor<3072x8192xbf16>
    %3495 = "ttir.permute"(%3493, %3494) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %3496 = "ttir.dot_general"(%3468, %3495) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %3497 = ttir.empty() : tensor<1x640x8192xbf16>
    %3498 = "ttir.reshape"(%3496, %3497) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3499 = ttir.empty() : tensor<1x640x8192xf32>
    %3500 = "ttir.typecast"(%3498, %3499) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3501 = ttir.empty() : tensor<1x640x8192xf32>
    %3502 = "ttir.multiply"(%3489, %3500, %3501) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3503 = ttir.empty() : tensor<1x640x8192xbf16>
    %3504 = "ttir.typecast"(%3502, %3503) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3505 = ttir.empty() : tensor<640x8192xbf16>
    %3506 = "ttir.reshape"(%3504, %3505) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %3507 = ttir.empty() : tensor<1x3072x8192xbf16>
    %3508 = "ttir.reshape"(%arg109, %3507) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %3509 = ttir.empty() : tensor<3072x8192xbf16>
    %3510 = "ttir.reshape"(%3508, %3509) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %3511 = ttir.empty() : tensor<8192x3072xbf16>
    %3512 = "ttir.permute"(%3510, %3511) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %3513 = "ttir.dot_general"(%3506, %3512) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %3514 = ttir.empty() : tensor<1x640x3072xbf16>
    %3515 = "ttir.reshape"(%3513, %3514) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3516 = ttir.empty() : tensor<1x640x3072xbf16>
    %3517 = "ttir.add"(%3426, %3515, %3516) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3518 = ttir.empty() : tensor<1x640x3072xf32>
    %3519 = "ttir.typecast"(%3517, %3518) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3520 = ttir.empty() : tensor<1x640x3072xf32>
    %3521 = "ttir.pow"(%3519, %5, %3520) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3522 = ttir.empty() : tensor<1x640xf32>
    %3523 = "ttir.sum"(%3521, %3522) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3524 = ttir.empty() : tensor<1x640xf32>
    %3525 = "ttir.multiply"(%3523, %4, %3524) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3526 = ttir.empty() : tensor<1x640x1xf32>
    %3527 = "ttir.reshape"(%3525, %3526) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3528 = ttir.empty() : tensor<1x640x1xf32>
    %3529 = "ttir.add"(%3527, %46, %3528) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3530 = ttir.empty() : tensor<1x640x1xf32>
    %3531 = "ttir.rsqrt"(%3529, %3530) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3532 = ttir.empty() : tensor<1x640xf32>
    %3533 = "ttir.reshape"(%3531, %3532) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3534 = ttir.empty() : tensor<1x640x1xf32>
    %3535 = "ttir.reshape"(%3533, %3534) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3536 = ttir.empty() : tensor<1x640x3072xf32>
    %3537 = "ttir.broadcast"(%3535, %3536) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3538 = ttir.empty() : tensor<1x640x3072xf32>
    %3539 = "ttir.multiply"(%3519, %3537, %3538) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3540 = ttir.empty() : tensor<1x640x3072xbf16>
    %3541 = "ttir.typecast"(%3539, %3540) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3542 = ttir.empty() : tensor<1x640x3072xf32>
    %3543 = "ttir.typecast"(%3541, %3542) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3544 = ttir.empty() : tensor<1x640x3072xf32>
    %3545 = "ttir.multiply"(%3289, %3543, %3544) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3546 = ttir.empty() : tensor<1x640x3072xbf16>
    %3547 = "ttir.typecast"(%3545, %3546) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3548 = ttir.empty() : tensor<640x3072xbf16>
    %3549 = "ttir.reshape"(%3547, %3548) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %3550 = ttir.empty() : tensor<1x1024x3072xbf16>
    %3551 = "ttir.reshape"(%arg108, %3550) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %3552 = ttir.empty() : tensor<1024x3072xbf16>
    %3553 = "ttir.reshape"(%3551, %3552) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %3554 = ttir.empty() : tensor<3072x1024xbf16>
    %3555 = "ttir.permute"(%3553, %3554) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %3556 = "ttir.dot_general"(%3549, %3555) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %3557 = ttir.empty() : tensor<1x640x8x128xbf16>
    %3558 = "ttir.reshape"(%3556, %3557) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %3559 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3560 = "ttir.permute"(%3558, %3559) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3561 = ttir.empty() : tensor<1x1x3072xbf16>
    %3562 = "ttir.reshape"(%arg125, %3561) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %3563 = ttir.empty() : tensor<3072xbf16>
    %3564 = "ttir.reshape"(%3562, %3563) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %3565 = ttir.empty() : tensor<3072xf32>
    %3566 = "ttir.typecast"(%3564, %3565) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %3567 = ttir.empty() : tensor<1x1x3072xf32>
    %3568 = "ttir.reshape"(%3566, %3567) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %3569 = ttir.empty() : tensor<1x640x3072xf32>
    %3570 = "ttir.broadcast"(%3568, %3569) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3571 = ttir.empty() : tensor<1x3072x3072xbf16>
    %3572 = "ttir.reshape"(%arg122, %3571) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %3573 = ttir.empty() : tensor<3072x3072xbf16>
    %3574 = "ttir.reshape"(%3572, %3573) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3575 = ttir.empty() : tensor<3072x3072xbf16>
    %3576 = "ttir.permute"(%3574, %3575) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3577 = "ttir.dot_general"(%3549, %3576) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %3578 = ttir.empty() : tensor<1x640x24x128xbf16>
    %3579 = "ttir.reshape"(%3577, %3578) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %3580 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3581 = "ttir.permute"(%3579, %3580) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3582 = ttir.empty() : tensor<1x24x640x128xf32>
    %3583 = "ttir.typecast"(%3581, %3582) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3584 = ttir.empty() : tensor<1x24x640x128xf32>
    %3585 = "ttir.multiply"(%3583, %125, %3584) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3586 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3587 = "ttir.typecast"(%3585, %3586) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3588 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3589 = "ttir.slice_static"(%3581, %3588) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3590 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3591 = "ttir.neg"(%3589, %3590) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3592 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3593 = "ttir.slice_static"(%3581, %3592) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3594 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3595 = "ttir.concat"(%3591, %3593, %3594) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3596 = ttir.empty() : tensor<1x24x640x128xf32>
    %3597 = "ttir.typecast"(%3595, %3596) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3598 = ttir.empty() : tensor<1x24x640x128xf32>
    %3599 = "ttir.multiply"(%3597, %153, %3598) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3600 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3601 = "ttir.typecast"(%3599, %3600) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3602 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3603 = "ttir.add"(%3587, %3601, %3602) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3604 = ttir.empty() : tensor<24x640x128xbf16>
    %3605 = "ttir.reshape"(%3603, %3604) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3606 = ttir.empty() : tensor<1x1024x3072xbf16>
    %3607 = "ttir.reshape"(%arg121, %3606) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %3608 = ttir.empty() : tensor<1024x3072xbf16>
    %3609 = "ttir.reshape"(%3607, %3608) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %3610 = ttir.empty() : tensor<3072x1024xbf16>
    %3611 = "ttir.permute"(%3609, %3610) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %3612 = "ttir.dot_general"(%3549, %3611) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %3613 = ttir.empty() : tensor<1x640x8x128xbf16>
    %3614 = "ttir.reshape"(%3612, %3613) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %3615 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3616 = "ttir.permute"(%3614, %3615) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3617 = ttir.empty() : tensor<1x8x640x128xf32>
    %3618 = "ttir.typecast"(%3616, %3617) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3619 = ttir.empty() : tensor<1x8x640x128xf32>
    %3620 = "ttir.multiply"(%3618, %178, %3619) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3621 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3622 = "ttir.typecast"(%3620, %3621) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3623 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3624 = "ttir.slice_static"(%3616, %3623) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3625 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3626 = "ttir.neg"(%3624, %3625) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3627 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3628 = "ttir.slice_static"(%3616, %3627) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3629 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3630 = "ttir.concat"(%3626, %3628, %3629) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3631 = ttir.empty() : tensor<1x8x640x128xf32>
    %3632 = "ttir.typecast"(%3630, %3631) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3633 = ttir.empty() : tensor<1x8x640x128xf32>
    %3634 = "ttir.multiply"(%3632, %196, %3633) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3635 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3636 = "ttir.typecast"(%3634, %3635) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3637 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3638 = "ttir.add"(%3622, %3636, %3637) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3639 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %3640 = "ttir.reshape"(%3638, %3639) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %3641 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %3642 = "ttir.broadcast"(%3640, %3641) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %3643 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3644 = "ttir.reshape"(%3642, %3643) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3645 = ttir.empty() : tensor<1x24x128x640xbf16>
    %3646 = "ttir.permute"(%3644, %3645) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %3647 = ttir.empty() : tensor<24x128x640xbf16>
    %3648 = "ttir.reshape"(%3646, %3647) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %3649 = "ttir.dot_general"(%3605, %3648) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %3650 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3651 = "ttir.reshape"(%3649, %3650) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3652 = ttir.empty() : tensor<1x24x640x640xf32>
    %3653 = "ttir.typecast"(%3651, %3652) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3654 = ttir.empty() : tensor<1x24x640x640xf32>
    %3655 = "ttir.multiply"(%3653, %221, %3654) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3656 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3657 = "ttir.typecast"(%3655, %3656) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3658 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3659 = "ttir.add"(%3657, %285, %3658) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3660 = ttir.empty() : tensor<1x24x640x640xf32>
    %3661 = "ttir.typecast"(%3659, %3660) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3662 = ttir.empty() : tensor<1x24x640xf32>
    %3663 = "ttir.max"(%3661, %3662) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %3664 = ttir.empty() : tensor<1x24x640x1xf32>
    %3665 = "ttir.reshape"(%3663, %3664) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %3666 = ttir.empty() : tensor<1x24x640x640xf32>
    %3667 = "ttir.broadcast"(%3665, %3666) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3668 = ttir.empty() : tensor<1x24x640x640xf32>
    %3669 = "ttir.subtract"(%3661, %3667, %3668) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3670 = ttir.empty() : tensor<1x24x640x640xf32>
    %3671 = "ttir.exp"(%3669, %3670) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3672 = ttir.empty() : tensor<1x24x640xf32>
    %3673 = "ttir.sum"(%3671, %3672) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %3674 = ttir.empty() : tensor<1x24x640x1xf32>
    %3675 = "ttir.reshape"(%3673, %3674) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %3676 = ttir.empty() : tensor<1x24x640x640xf32>
    %3677 = "ttir.broadcast"(%3675, %3676) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3678 = ttir.empty() : tensor<1x24x640x640xf32>
    %3679 = "ttir.div"(%3671, %3677, %3678) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3680 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3681 = "ttir.typecast"(%3679, %3680) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3682 = ttir.empty() : tensor<24x640x640xbf16>
    %3683 = "ttir.reshape"(%3681, %3682) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %3684 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %3685 = "ttir.reshape"(%3560, %3684) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %3686 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %3687 = "ttir.broadcast"(%3685, %3686) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %3688 = ttir.empty() : tensor<24x640x128xbf16>
    %3689 = "ttir.reshape"(%3687, %3688) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3690 = "ttir.dot_general"(%3683, %3689) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3691 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3692 = "ttir.reshape"(%3690, %3691) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3693 = ttir.empty() : tensor<1x640x24x128xbf16>
    %3694 = "ttir.permute"(%3692, %3693) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %3695 = ttir.empty() : tensor<640x3072xbf16>
    %3696 = "ttir.reshape"(%3694, %3695) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %3697 = ttir.empty() : tensor<1x3072x3072xbf16>
    %3698 = "ttir.reshape"(%arg120, %3697) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %3699 = ttir.empty() : tensor<3072x3072xbf16>
    %3700 = "ttir.reshape"(%3698, %3699) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3701 = ttir.empty() : tensor<3072x3072xbf16>
    %3702 = "ttir.permute"(%3700, %3701) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3703 = "ttir.dot_general"(%3696, %3702) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %3704 = ttir.empty() : tensor<1x640x3072xbf16>
    %3705 = "ttir.reshape"(%3703, %3704) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3706 = ttir.empty() : tensor<1x640x3072xbf16>
    %3707 = "ttir.add"(%3517, %3705, %3706) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3708 = ttir.empty() : tensor<1x1x3072xbf16>
    %3709 = "ttir.reshape"(%arg123, %3708) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %3710 = ttir.empty() : tensor<3072xbf16>
    %3711 = "ttir.reshape"(%3709, %3710) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %3712 = ttir.empty() : tensor<3072xf32>
    %3713 = "ttir.typecast"(%3711, %3712) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %3714 = ttir.empty() : tensor<1x1x3072xf32>
    %3715 = "ttir.reshape"(%3713, %3714) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %3716 = ttir.empty() : tensor<1x640x3072xf32>
    %3717 = "ttir.broadcast"(%3715, %3716) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3718 = ttir.empty() : tensor<1x640x3072xf32>
    %3719 = "ttir.typecast"(%3707, %3718) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3720 = ttir.empty() : tensor<1x640x3072xf32>
    %3721 = "ttir.pow"(%3719, %5, %3720) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3722 = ttir.empty() : tensor<1x640xf32>
    %3723 = "ttir.sum"(%3721, %3722) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3724 = ttir.empty() : tensor<1x640xf32>
    %3725 = "ttir.multiply"(%3723, %4, %3724) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3726 = ttir.empty() : tensor<1x640x1xf32>
    %3727 = "ttir.reshape"(%3725, %3726) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3728 = ttir.empty() : tensor<1x640x1xf32>
    %3729 = "ttir.add"(%3727, %46, %3728) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3730 = ttir.empty() : tensor<1x640x1xf32>
    %3731 = "ttir.rsqrt"(%3729, %3730) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3732 = ttir.empty() : tensor<1x640xf32>
    %3733 = "ttir.reshape"(%3731, %3732) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3734 = ttir.empty() : tensor<1x640x1xf32>
    %3735 = "ttir.reshape"(%3733, %3734) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3736 = ttir.empty() : tensor<1x640x3072xf32>
    %3737 = "ttir.broadcast"(%3735, %3736) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3738 = ttir.empty() : tensor<1x640x3072xf32>
    %3739 = "ttir.multiply"(%3719, %3737, %3738) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3740 = ttir.empty() : tensor<1x640x3072xbf16>
    %3741 = "ttir.typecast"(%3739, %3740) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3742 = ttir.empty() : tensor<1x640x3072xf32>
    %3743 = "ttir.typecast"(%3741, %3742) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3744 = ttir.empty() : tensor<1x640x3072xf32>
    %3745 = "ttir.multiply"(%3717, %3743, %3744) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3746 = ttir.empty() : tensor<1x640x3072xbf16>
    %3747 = "ttir.typecast"(%3745, %3746) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3748 = ttir.empty() : tensor<640x3072xbf16>
    %3749 = "ttir.reshape"(%3747, %3748) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %3750 = ttir.empty() : tensor<1x8192x3072xbf16>
    %3751 = "ttir.reshape"(%arg124, %3750) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %3752 = ttir.empty() : tensor<8192x3072xbf16>
    %3753 = "ttir.reshape"(%3751, %3752) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %3754 = ttir.empty() : tensor<3072x8192xbf16>
    %3755 = "ttir.permute"(%3753, %3754) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %3756 = "ttir.dot_general"(%3749, %3755) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %3757 = ttir.empty() : tensor<1x640x8192xbf16>
    %3758 = "ttir.reshape"(%3756, %3757) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3759 = ttir.empty() : tensor<1x640x8192xf32>
    %3760 = "ttir.typecast"(%3758, %3759) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3761 = ttir.empty() : tensor<1x640x8192xbf16>
    %3762 = "ttir.sigmoid"(%3758, %3761) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3763 = ttir.empty() : tensor<1x640x8192xf32>
    %3764 = "ttir.typecast"(%3762, %3763) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3765 = ttir.empty() : tensor<1x640x8192xf32>
    %3766 = "ttir.multiply"(%3760, %3764, %3765) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3767 = ttir.empty() : tensor<1x640x8192xbf16>
    %3768 = "ttir.typecast"(%3766, %3767) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3769 = ttir.empty() : tensor<1x640x8192xf32>
    %3770 = "ttir.typecast"(%3768, %3769) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3771 = ttir.empty() : tensor<1x8192x3072xbf16>
    %3772 = "ttir.reshape"(%arg119, %3771) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %3773 = ttir.empty() : tensor<8192x3072xbf16>
    %3774 = "ttir.reshape"(%3772, %3773) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %3775 = ttir.empty() : tensor<3072x8192xbf16>
    %3776 = "ttir.permute"(%3774, %3775) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %3777 = "ttir.dot_general"(%3749, %3776) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %3778 = ttir.empty() : tensor<1x640x8192xbf16>
    %3779 = "ttir.reshape"(%3777, %3778) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3780 = ttir.empty() : tensor<1x640x8192xf32>
    %3781 = "ttir.typecast"(%3779, %3780) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3782 = ttir.empty() : tensor<1x640x8192xf32>
    %3783 = "ttir.multiply"(%3770, %3781, %3782) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %3784 = ttir.empty() : tensor<1x640x8192xbf16>
    %3785 = "ttir.typecast"(%3783, %3784) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %3786 = ttir.empty() : tensor<640x8192xbf16>
    %3787 = "ttir.reshape"(%3785, %3786) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %3788 = ttir.empty() : tensor<1x3072x8192xbf16>
    %3789 = "ttir.reshape"(%arg118, %3788) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %3790 = ttir.empty() : tensor<3072x8192xbf16>
    %3791 = "ttir.reshape"(%3789, %3790) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %3792 = ttir.empty() : tensor<8192x3072xbf16>
    %3793 = "ttir.permute"(%3791, %3792) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %3794 = "ttir.dot_general"(%3787, %3793) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %3795 = ttir.empty() : tensor<1x640x3072xbf16>
    %3796 = "ttir.reshape"(%3794, %3795) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3797 = ttir.empty() : tensor<1x640x3072xbf16>
    %3798 = "ttir.add"(%3707, %3796, %3797) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3799 = ttir.empty() : tensor<1x640x3072xf32>
    %3800 = "ttir.typecast"(%3798, %3799) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3801 = ttir.empty() : tensor<1x640x3072xf32>
    %3802 = "ttir.pow"(%3800, %5, %3801) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3803 = ttir.empty() : tensor<1x640xf32>
    %3804 = "ttir.sum"(%3802, %3803) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3805 = ttir.empty() : tensor<1x640xf32>
    %3806 = "ttir.multiply"(%3804, %4, %3805) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3807 = ttir.empty() : tensor<1x640x1xf32>
    %3808 = "ttir.reshape"(%3806, %3807) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3809 = ttir.empty() : tensor<1x640x1xf32>
    %3810 = "ttir.add"(%3808, %46, %3809) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3811 = ttir.empty() : tensor<1x640x1xf32>
    %3812 = "ttir.rsqrt"(%3810, %3811) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3813 = ttir.empty() : tensor<1x640xf32>
    %3814 = "ttir.reshape"(%3812, %3813) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %3815 = ttir.empty() : tensor<1x640x1xf32>
    %3816 = "ttir.reshape"(%3814, %3815) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %3817 = ttir.empty() : tensor<1x640x3072xf32>
    %3818 = "ttir.broadcast"(%3816, %3817) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3819 = ttir.empty() : tensor<1x640x3072xf32>
    %3820 = "ttir.multiply"(%3800, %3818, %3819) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3821 = ttir.empty() : tensor<1x640x3072xbf16>
    %3822 = "ttir.typecast"(%3820, %3821) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3823 = ttir.empty() : tensor<1x640x3072xf32>
    %3824 = "ttir.typecast"(%3822, %3823) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3825 = ttir.empty() : tensor<1x640x3072xf32>
    %3826 = "ttir.multiply"(%3570, %3824, %3825) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3827 = ttir.empty() : tensor<1x640x3072xbf16>
    %3828 = "ttir.typecast"(%3826, %3827) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3829 = ttir.empty() : tensor<640x3072xbf16>
    %3830 = "ttir.reshape"(%3828, %3829) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %3831 = ttir.empty() : tensor<1x1024x3072xbf16>
    %3832 = "ttir.reshape"(%arg117, %3831) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %3833 = ttir.empty() : tensor<1024x3072xbf16>
    %3834 = "ttir.reshape"(%3832, %3833) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %3835 = ttir.empty() : tensor<3072x1024xbf16>
    %3836 = "ttir.permute"(%3834, %3835) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %3837 = "ttir.dot_general"(%3830, %3836) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %3838 = ttir.empty() : tensor<1x640x8x128xbf16>
    %3839 = "ttir.reshape"(%3837, %3838) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %3840 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3841 = "ttir.permute"(%3839, %3840) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3842 = ttir.empty() : tensor<1x1x3072xbf16>
    %3843 = "ttir.reshape"(%arg134, %3842) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %3844 = ttir.empty() : tensor<3072xbf16>
    %3845 = "ttir.reshape"(%3843, %3844) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %3846 = ttir.empty() : tensor<3072xf32>
    %3847 = "ttir.typecast"(%3845, %3846) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %3848 = ttir.empty() : tensor<1x1x3072xf32>
    %3849 = "ttir.reshape"(%3847, %3848) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %3850 = ttir.empty() : tensor<1x640x3072xf32>
    %3851 = "ttir.broadcast"(%3849, %3850) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3852 = ttir.empty() : tensor<1x3072x3072xbf16>
    %3853 = "ttir.reshape"(%arg131, %3852) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %3854 = ttir.empty() : tensor<3072x3072xbf16>
    %3855 = "ttir.reshape"(%3853, %3854) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3856 = ttir.empty() : tensor<3072x3072xbf16>
    %3857 = "ttir.permute"(%3855, %3856) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3858 = "ttir.dot_general"(%3830, %3857) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %3859 = ttir.empty() : tensor<1x640x24x128xbf16>
    %3860 = "ttir.reshape"(%3858, %3859) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %3861 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3862 = "ttir.permute"(%3860, %3861) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3863 = ttir.empty() : tensor<1x24x640x128xf32>
    %3864 = "ttir.typecast"(%3862, %3863) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3865 = ttir.empty() : tensor<1x24x640x128xf32>
    %3866 = "ttir.multiply"(%3864, %125, %3865) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3867 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3868 = "ttir.typecast"(%3866, %3867) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3869 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3870 = "ttir.slice_static"(%3862, %3869) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3871 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3872 = "ttir.neg"(%3870, %3871) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3873 = ttir.empty() : tensor<1x24x640x64xbf16>
    %3874 = "ttir.slice_static"(%3862, %3873) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %3875 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3876 = "ttir.concat"(%3872, %3874, %3875) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3877 = ttir.empty() : tensor<1x24x640x128xf32>
    %3878 = "ttir.typecast"(%3876, %3877) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3879 = ttir.empty() : tensor<1x24x640x128xf32>
    %3880 = "ttir.multiply"(%3878, %153, %3879) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %3881 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3882 = "ttir.typecast"(%3880, %3881) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3883 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3884 = "ttir.add"(%3868, %3882, %3883) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3885 = ttir.empty() : tensor<24x640x128xbf16>
    %3886 = "ttir.reshape"(%3884, %3885) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3887 = ttir.empty() : tensor<1x1024x3072xbf16>
    %3888 = "ttir.reshape"(%arg130, %3887) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %3889 = ttir.empty() : tensor<1024x3072xbf16>
    %3890 = "ttir.reshape"(%3888, %3889) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %3891 = ttir.empty() : tensor<3072x1024xbf16>
    %3892 = "ttir.permute"(%3890, %3891) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %3893 = "ttir.dot_general"(%3830, %3892) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %3894 = ttir.empty() : tensor<1x640x8x128xbf16>
    %3895 = "ttir.reshape"(%3893, %3894) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %3896 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3897 = "ttir.permute"(%3895, %3896) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3898 = ttir.empty() : tensor<1x8x640x128xf32>
    %3899 = "ttir.typecast"(%3897, %3898) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3900 = ttir.empty() : tensor<1x8x640x128xf32>
    %3901 = "ttir.multiply"(%3899, %178, %3900) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3902 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3903 = "ttir.typecast"(%3901, %3902) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3904 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3905 = "ttir.slice_static"(%3897, %3904) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3906 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3907 = "ttir.neg"(%3905, %3906) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3908 = ttir.empty() : tensor<1x8x640x64xbf16>
    %3909 = "ttir.slice_static"(%3897, %3908) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %3910 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3911 = "ttir.concat"(%3907, %3909, %3910) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3912 = ttir.empty() : tensor<1x8x640x128xf32>
    %3913 = "ttir.typecast"(%3911, %3912) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3914 = ttir.empty() : tensor<1x8x640x128xf32>
    %3915 = "ttir.multiply"(%3913, %196, %3914) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %3916 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3917 = "ttir.typecast"(%3915, %3916) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3918 = ttir.empty() : tensor<1x8x640x128xbf16>
    %3919 = "ttir.add"(%3903, %3917, %3918) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %3920 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %3921 = "ttir.reshape"(%3919, %3920) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %3922 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %3923 = "ttir.broadcast"(%3921, %3922) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %3924 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3925 = "ttir.reshape"(%3923, %3924) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3926 = ttir.empty() : tensor<1x24x128x640xbf16>
    %3927 = "ttir.permute"(%3925, %3926) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %3928 = ttir.empty() : tensor<24x128x640xbf16>
    %3929 = "ttir.reshape"(%3927, %3928) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %3930 = "ttir.dot_general"(%3886, %3929) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %3931 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3932 = "ttir.reshape"(%3930, %3931) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3933 = ttir.empty() : tensor<1x24x640x640xf32>
    %3934 = "ttir.typecast"(%3932, %3933) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3935 = ttir.empty() : tensor<1x24x640x640xf32>
    %3936 = "ttir.multiply"(%3934, %221, %3935) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3937 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3938 = "ttir.typecast"(%3936, %3937) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3939 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3940 = "ttir.add"(%3938, %285, %3939) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3941 = ttir.empty() : tensor<1x24x640x640xf32>
    %3942 = "ttir.typecast"(%3940, %3941) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3943 = ttir.empty() : tensor<1x24x640xf32>
    %3944 = "ttir.max"(%3942, %3943) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %3945 = ttir.empty() : tensor<1x24x640x1xf32>
    %3946 = "ttir.reshape"(%3944, %3945) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %3947 = ttir.empty() : tensor<1x24x640x640xf32>
    %3948 = "ttir.broadcast"(%3946, %3947) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3949 = ttir.empty() : tensor<1x24x640x640xf32>
    %3950 = "ttir.subtract"(%3942, %3948, %3949) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3951 = ttir.empty() : tensor<1x24x640x640xf32>
    %3952 = "ttir.exp"(%3950, %3951) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3953 = ttir.empty() : tensor<1x24x640xf32>
    %3954 = "ttir.sum"(%3952, %3953) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %3955 = ttir.empty() : tensor<1x24x640x1xf32>
    %3956 = "ttir.reshape"(%3954, %3955) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %3957 = ttir.empty() : tensor<1x24x640x640xf32>
    %3958 = "ttir.broadcast"(%3956, %3957) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3959 = ttir.empty() : tensor<1x24x640x640xf32>
    %3960 = "ttir.div"(%3952, %3958, %3959) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %3961 = ttir.empty() : tensor<1x24x640x640xbf16>
    %3962 = "ttir.typecast"(%3960, %3961) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %3963 = ttir.empty() : tensor<24x640x640xbf16>
    %3964 = "ttir.reshape"(%3962, %3963) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %3965 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %3966 = "ttir.reshape"(%3841, %3965) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %3967 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %3968 = "ttir.broadcast"(%3966, %3967) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %3969 = ttir.empty() : tensor<24x640x128xbf16>
    %3970 = "ttir.reshape"(%3968, %3969) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3971 = "ttir.dot_general"(%3964, %3970) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %3972 = ttir.empty() : tensor<1x24x640x128xbf16>
    %3973 = "ttir.reshape"(%3971, %3972) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %3974 = ttir.empty() : tensor<1x640x24x128xbf16>
    %3975 = "ttir.permute"(%3973, %3974) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %3976 = ttir.empty() : tensor<640x3072xbf16>
    %3977 = "ttir.reshape"(%3975, %3976) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %3978 = ttir.empty() : tensor<1x3072x3072xbf16>
    %3979 = "ttir.reshape"(%arg129, %3978) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %3980 = ttir.empty() : tensor<3072x3072xbf16>
    %3981 = "ttir.reshape"(%3979, %3980) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3982 = ttir.empty() : tensor<3072x3072xbf16>
    %3983 = "ttir.permute"(%3981, %3982) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3984 = "ttir.dot_general"(%3977, %3983) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %3985 = ttir.empty() : tensor<1x640x3072xbf16>
    %3986 = "ttir.reshape"(%3984, %3985) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3987 = ttir.empty() : tensor<1x640x3072xbf16>
    %3988 = "ttir.add"(%3798, %3986, %3987) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %3989 = ttir.empty() : tensor<1x1x3072xbf16>
    %3990 = "ttir.reshape"(%arg132, %3989) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %3991 = ttir.empty() : tensor<3072xbf16>
    %3992 = "ttir.reshape"(%3990, %3991) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %3993 = ttir.empty() : tensor<3072xf32>
    %3994 = "ttir.typecast"(%3992, %3993) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %3995 = ttir.empty() : tensor<1x1x3072xf32>
    %3996 = "ttir.reshape"(%3994, %3995) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %3997 = ttir.empty() : tensor<1x640x3072xf32>
    %3998 = "ttir.broadcast"(%3996, %3997) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %3999 = ttir.empty() : tensor<1x640x3072xf32>
    %4000 = "ttir.typecast"(%3988, %3999) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4001 = ttir.empty() : tensor<1x640x3072xf32>
    %4002 = "ttir.pow"(%4000, %5, %4001) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4003 = ttir.empty() : tensor<1x640xf32>
    %4004 = "ttir.sum"(%4002, %4003) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4005 = ttir.empty() : tensor<1x640xf32>
    %4006 = "ttir.multiply"(%4004, %4, %4005) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4007 = ttir.empty() : tensor<1x640x1xf32>
    %4008 = "ttir.reshape"(%4006, %4007) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4009 = ttir.empty() : tensor<1x640x1xf32>
    %4010 = "ttir.add"(%4008, %46, %4009) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4011 = ttir.empty() : tensor<1x640x1xf32>
    %4012 = "ttir.rsqrt"(%4010, %4011) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4013 = ttir.empty() : tensor<1x640xf32>
    %4014 = "ttir.reshape"(%4012, %4013) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4015 = ttir.empty() : tensor<1x640x1xf32>
    %4016 = "ttir.reshape"(%4014, %4015) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4017 = ttir.empty() : tensor<1x640x3072xf32>
    %4018 = "ttir.broadcast"(%4016, %4017) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4019 = ttir.empty() : tensor<1x640x3072xf32>
    %4020 = "ttir.multiply"(%4000, %4018, %4019) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4021 = ttir.empty() : tensor<1x640x3072xbf16>
    %4022 = "ttir.typecast"(%4020, %4021) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4023 = ttir.empty() : tensor<1x640x3072xf32>
    %4024 = "ttir.typecast"(%4022, %4023) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4025 = ttir.empty() : tensor<1x640x3072xf32>
    %4026 = "ttir.multiply"(%3998, %4024, %4025) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4027 = ttir.empty() : tensor<1x640x3072xbf16>
    %4028 = "ttir.typecast"(%4026, %4027) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4029 = ttir.empty() : tensor<640x3072xbf16>
    %4030 = "ttir.reshape"(%4028, %4029) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4031 = ttir.empty() : tensor<1x8192x3072xbf16>
    %4032 = "ttir.reshape"(%arg133, %4031) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %4033 = ttir.empty() : tensor<8192x3072xbf16>
    %4034 = "ttir.reshape"(%4032, %4033) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4035 = ttir.empty() : tensor<3072x8192xbf16>
    %4036 = "ttir.permute"(%4034, %4035) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4037 = "ttir.dot_general"(%4030, %4036) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %4038 = ttir.empty() : tensor<1x640x8192xbf16>
    %4039 = "ttir.reshape"(%4037, %4038) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4040 = ttir.empty() : tensor<1x640x8192xf32>
    %4041 = "ttir.typecast"(%4039, %4040) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4042 = ttir.empty() : tensor<1x640x8192xbf16>
    %4043 = "ttir.sigmoid"(%4039, %4042) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4044 = ttir.empty() : tensor<1x640x8192xf32>
    %4045 = "ttir.typecast"(%4043, %4044) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4046 = ttir.empty() : tensor<1x640x8192xf32>
    %4047 = "ttir.multiply"(%4041, %4045, %4046) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4048 = ttir.empty() : tensor<1x640x8192xbf16>
    %4049 = "ttir.typecast"(%4047, %4048) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4050 = ttir.empty() : tensor<1x640x8192xf32>
    %4051 = "ttir.typecast"(%4049, %4050) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4052 = ttir.empty() : tensor<1x8192x3072xbf16>
    %4053 = "ttir.reshape"(%arg128, %4052) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %4054 = ttir.empty() : tensor<8192x3072xbf16>
    %4055 = "ttir.reshape"(%4053, %4054) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4056 = ttir.empty() : tensor<3072x8192xbf16>
    %4057 = "ttir.permute"(%4055, %4056) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4058 = "ttir.dot_general"(%4030, %4057) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %4059 = ttir.empty() : tensor<1x640x8192xbf16>
    %4060 = "ttir.reshape"(%4058, %4059) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4061 = ttir.empty() : tensor<1x640x8192xf32>
    %4062 = "ttir.typecast"(%4060, %4061) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4063 = ttir.empty() : tensor<1x640x8192xf32>
    %4064 = "ttir.multiply"(%4051, %4062, %4063) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4065 = ttir.empty() : tensor<1x640x8192xbf16>
    %4066 = "ttir.typecast"(%4064, %4065) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4067 = ttir.empty() : tensor<640x8192xbf16>
    %4068 = "ttir.reshape"(%4066, %4067) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %4069 = ttir.empty() : tensor<1x3072x8192xbf16>
    %4070 = "ttir.reshape"(%arg127, %4069) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %4071 = ttir.empty() : tensor<3072x8192xbf16>
    %4072 = "ttir.reshape"(%4070, %4071) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4073 = ttir.empty() : tensor<8192x3072xbf16>
    %4074 = "ttir.permute"(%4072, %4073) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4075 = "ttir.dot_general"(%4068, %4074) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %4076 = ttir.empty() : tensor<1x640x3072xbf16>
    %4077 = "ttir.reshape"(%4075, %4076) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4078 = ttir.empty() : tensor<1x640x3072xbf16>
    %4079 = "ttir.add"(%3988, %4077, %4078) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4080 = ttir.empty() : tensor<1x640x3072xf32>
    %4081 = "ttir.typecast"(%4079, %4080) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4082 = ttir.empty() : tensor<1x640x3072xf32>
    %4083 = "ttir.pow"(%4081, %5, %4082) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4084 = ttir.empty() : tensor<1x640xf32>
    %4085 = "ttir.sum"(%4083, %4084) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4086 = ttir.empty() : tensor<1x640xf32>
    %4087 = "ttir.multiply"(%4085, %4, %4086) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4088 = ttir.empty() : tensor<1x640x1xf32>
    %4089 = "ttir.reshape"(%4087, %4088) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4090 = ttir.empty() : tensor<1x640x1xf32>
    %4091 = "ttir.add"(%4089, %46, %4090) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4092 = ttir.empty() : tensor<1x640x1xf32>
    %4093 = "ttir.rsqrt"(%4091, %4092) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4094 = ttir.empty() : tensor<1x640xf32>
    %4095 = "ttir.reshape"(%4093, %4094) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4096 = ttir.empty() : tensor<1x640x1xf32>
    %4097 = "ttir.reshape"(%4095, %4096) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4098 = ttir.empty() : tensor<1x640x3072xf32>
    %4099 = "ttir.broadcast"(%4097, %4098) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4100 = ttir.empty() : tensor<1x640x3072xf32>
    %4101 = "ttir.multiply"(%4081, %4099, %4100) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4102 = ttir.empty() : tensor<1x640x3072xbf16>
    %4103 = "ttir.typecast"(%4101, %4102) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4104 = ttir.empty() : tensor<1x640x3072xf32>
    %4105 = "ttir.typecast"(%4103, %4104) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4106 = ttir.empty() : tensor<1x640x3072xf32>
    %4107 = "ttir.multiply"(%3851, %4105, %4106) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4108 = ttir.empty() : tensor<1x640x3072xbf16>
    %4109 = "ttir.typecast"(%4107, %4108) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4110 = ttir.empty() : tensor<640x3072xbf16>
    %4111 = "ttir.reshape"(%4109, %4110) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4112 = ttir.empty() : tensor<1x1024x3072xbf16>
    %4113 = "ttir.reshape"(%arg126, %4112) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %4114 = ttir.empty() : tensor<1024x3072xbf16>
    %4115 = "ttir.reshape"(%4113, %4114) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %4116 = ttir.empty() : tensor<3072x1024xbf16>
    %4117 = "ttir.permute"(%4115, %4116) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %4118 = "ttir.dot_general"(%4111, %4117) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %4119 = ttir.empty() : tensor<1x640x8x128xbf16>
    %4120 = "ttir.reshape"(%4118, %4119) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %4121 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4122 = "ttir.permute"(%4120, %4121) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4123 = ttir.empty() : tensor<1x1x3072xbf16>
    %4124 = "ttir.reshape"(%arg143, %4123) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %4125 = ttir.empty() : tensor<3072xbf16>
    %4126 = "ttir.reshape"(%4124, %4125) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %4127 = ttir.empty() : tensor<3072xf32>
    %4128 = "ttir.typecast"(%4126, %4127) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %4129 = ttir.empty() : tensor<1x1x3072xf32>
    %4130 = "ttir.reshape"(%4128, %4129) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %4131 = ttir.empty() : tensor<1x640x3072xf32>
    %4132 = "ttir.broadcast"(%4130, %4131) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4133 = ttir.empty() : tensor<1x3072x3072xbf16>
    %4134 = "ttir.reshape"(%arg140, %4133) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %4135 = ttir.empty() : tensor<3072x3072xbf16>
    %4136 = "ttir.reshape"(%4134, %4135) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4137 = ttir.empty() : tensor<3072x3072xbf16>
    %4138 = "ttir.permute"(%4136, %4137) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4139 = "ttir.dot_general"(%4111, %4138) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %4140 = ttir.empty() : tensor<1x640x24x128xbf16>
    %4141 = "ttir.reshape"(%4139, %4140) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %4142 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4143 = "ttir.permute"(%4141, %4142) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4144 = ttir.empty() : tensor<1x24x640x128xf32>
    %4145 = "ttir.typecast"(%4143, %4144) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4146 = ttir.empty() : tensor<1x24x640x128xf32>
    %4147 = "ttir.multiply"(%4145, %125, %4146) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4148 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4149 = "ttir.typecast"(%4147, %4148) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4150 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4151 = "ttir.slice_static"(%4143, %4150) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4152 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4153 = "ttir.neg"(%4151, %4152) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4154 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4155 = "ttir.slice_static"(%4143, %4154) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4156 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4157 = "ttir.concat"(%4153, %4155, %4156) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4158 = ttir.empty() : tensor<1x24x640x128xf32>
    %4159 = "ttir.typecast"(%4157, %4158) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4160 = ttir.empty() : tensor<1x24x640x128xf32>
    %4161 = "ttir.multiply"(%4159, %153, %4160) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4162 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4163 = "ttir.typecast"(%4161, %4162) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4164 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4165 = "ttir.add"(%4149, %4163, %4164) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4166 = ttir.empty() : tensor<24x640x128xbf16>
    %4167 = "ttir.reshape"(%4165, %4166) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %4168 = ttir.empty() : tensor<1x1024x3072xbf16>
    %4169 = "ttir.reshape"(%arg139, %4168) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %4170 = ttir.empty() : tensor<1024x3072xbf16>
    %4171 = "ttir.reshape"(%4169, %4170) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %4172 = ttir.empty() : tensor<3072x1024xbf16>
    %4173 = "ttir.permute"(%4171, %4172) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %4174 = "ttir.dot_general"(%4111, %4173) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %4175 = ttir.empty() : tensor<1x640x8x128xbf16>
    %4176 = "ttir.reshape"(%4174, %4175) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %4177 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4178 = "ttir.permute"(%4176, %4177) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4179 = ttir.empty() : tensor<1x8x640x128xf32>
    %4180 = "ttir.typecast"(%4178, %4179) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4181 = ttir.empty() : tensor<1x8x640x128xf32>
    %4182 = "ttir.multiply"(%4180, %178, %4181) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4183 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4184 = "ttir.typecast"(%4182, %4183) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4185 = ttir.empty() : tensor<1x8x640x64xbf16>
    %4186 = "ttir.slice_static"(%4178, %4185) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %4187 = ttir.empty() : tensor<1x8x640x64xbf16>
    %4188 = "ttir.neg"(%4186, %4187) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %4189 = ttir.empty() : tensor<1x8x640x64xbf16>
    %4190 = "ttir.slice_static"(%4178, %4189) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %4191 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4192 = "ttir.concat"(%4188, %4190, %4191) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4193 = ttir.empty() : tensor<1x8x640x128xf32>
    %4194 = "ttir.typecast"(%4192, %4193) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4195 = ttir.empty() : tensor<1x8x640x128xf32>
    %4196 = "ttir.multiply"(%4194, %196, %4195) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4197 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4198 = "ttir.typecast"(%4196, %4197) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4199 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4200 = "ttir.add"(%4184, %4198, %4199) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4201 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %4202 = "ttir.reshape"(%4200, %4201) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %4203 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %4204 = "ttir.broadcast"(%4202, %4203) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %4205 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4206 = "ttir.reshape"(%4204, %4205) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4207 = ttir.empty() : tensor<1x24x128x640xbf16>
    %4208 = "ttir.permute"(%4206, %4207) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %4209 = ttir.empty() : tensor<24x128x640xbf16>
    %4210 = "ttir.reshape"(%4208, %4209) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %4211 = "ttir.dot_general"(%4167, %4210) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %4212 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4213 = "ttir.reshape"(%4211, %4212) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4214 = ttir.empty() : tensor<1x24x640x640xf32>
    %4215 = "ttir.typecast"(%4213, %4214) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4216 = ttir.empty() : tensor<1x24x640x640xf32>
    %4217 = "ttir.multiply"(%4215, %221, %4216) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4218 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4219 = "ttir.typecast"(%4217, %4218) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4220 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4221 = "ttir.add"(%4219, %285, %4220) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4222 = ttir.empty() : tensor<1x24x640x640xf32>
    %4223 = "ttir.typecast"(%4221, %4222) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4224 = ttir.empty() : tensor<1x24x640xf32>
    %4225 = "ttir.max"(%4223, %4224) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %4226 = ttir.empty() : tensor<1x24x640x1xf32>
    %4227 = "ttir.reshape"(%4225, %4226) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %4228 = ttir.empty() : tensor<1x24x640x640xf32>
    %4229 = "ttir.broadcast"(%4227, %4228) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4230 = ttir.empty() : tensor<1x24x640x640xf32>
    %4231 = "ttir.subtract"(%4223, %4229, %4230) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4232 = ttir.empty() : tensor<1x24x640x640xf32>
    %4233 = "ttir.exp"(%4231, %4232) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4234 = ttir.empty() : tensor<1x24x640xf32>
    %4235 = "ttir.sum"(%4233, %4234) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %4236 = ttir.empty() : tensor<1x24x640x1xf32>
    %4237 = "ttir.reshape"(%4235, %4236) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %4238 = ttir.empty() : tensor<1x24x640x640xf32>
    %4239 = "ttir.broadcast"(%4237, %4238) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4240 = ttir.empty() : tensor<1x24x640x640xf32>
    %4241 = "ttir.div"(%4233, %4239, %4240) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4242 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4243 = "ttir.typecast"(%4241, %4242) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4244 = ttir.empty() : tensor<24x640x640xbf16>
    %4245 = "ttir.reshape"(%4243, %4244) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %4246 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %4247 = "ttir.reshape"(%4122, %4246) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %4248 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %4249 = "ttir.broadcast"(%4247, %4248) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %4250 = ttir.empty() : tensor<24x640x128xbf16>
    %4251 = "ttir.reshape"(%4249, %4250) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %4252 = "ttir.dot_general"(%4245, %4251) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %4253 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4254 = "ttir.reshape"(%4252, %4253) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4255 = ttir.empty() : tensor<1x640x24x128xbf16>
    %4256 = "ttir.permute"(%4254, %4255) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %4257 = ttir.empty() : tensor<640x3072xbf16>
    %4258 = "ttir.reshape"(%4256, %4257) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4259 = ttir.empty() : tensor<1x3072x3072xbf16>
    %4260 = "ttir.reshape"(%arg138, %4259) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %4261 = ttir.empty() : tensor<3072x3072xbf16>
    %4262 = "ttir.reshape"(%4260, %4261) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4263 = ttir.empty() : tensor<3072x3072xbf16>
    %4264 = "ttir.permute"(%4262, %4263) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4265 = "ttir.dot_general"(%4258, %4264) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %4266 = ttir.empty() : tensor<1x640x3072xbf16>
    %4267 = "ttir.reshape"(%4265, %4266) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4268 = ttir.empty() : tensor<1x640x3072xbf16>
    %4269 = "ttir.add"(%4079, %4267, %4268) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4270 = ttir.empty() : tensor<1x1x3072xbf16>
    %4271 = "ttir.reshape"(%arg141, %4270) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %4272 = ttir.empty() : tensor<3072xbf16>
    %4273 = "ttir.reshape"(%4271, %4272) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %4274 = ttir.empty() : tensor<3072xf32>
    %4275 = "ttir.typecast"(%4273, %4274) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %4276 = ttir.empty() : tensor<1x1x3072xf32>
    %4277 = "ttir.reshape"(%4275, %4276) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %4278 = ttir.empty() : tensor<1x640x3072xf32>
    %4279 = "ttir.broadcast"(%4277, %4278) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4280 = ttir.empty() : tensor<1x640x3072xf32>
    %4281 = "ttir.typecast"(%4269, %4280) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4282 = ttir.empty() : tensor<1x640x3072xf32>
    %4283 = "ttir.pow"(%4281, %5, %4282) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4284 = ttir.empty() : tensor<1x640xf32>
    %4285 = "ttir.sum"(%4283, %4284) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4286 = ttir.empty() : tensor<1x640xf32>
    %4287 = "ttir.multiply"(%4285, %4, %4286) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4288 = ttir.empty() : tensor<1x640x1xf32>
    %4289 = "ttir.reshape"(%4287, %4288) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4290 = ttir.empty() : tensor<1x640x1xf32>
    %4291 = "ttir.add"(%4289, %46, %4290) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4292 = ttir.empty() : tensor<1x640x1xf32>
    %4293 = "ttir.rsqrt"(%4291, %4292) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4294 = ttir.empty() : tensor<1x640xf32>
    %4295 = "ttir.reshape"(%4293, %4294) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4296 = ttir.empty() : tensor<1x640x1xf32>
    %4297 = "ttir.reshape"(%4295, %4296) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4298 = ttir.empty() : tensor<1x640x3072xf32>
    %4299 = "ttir.broadcast"(%4297, %4298) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4300 = ttir.empty() : tensor<1x640x3072xf32>
    %4301 = "ttir.multiply"(%4281, %4299, %4300) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4302 = ttir.empty() : tensor<1x640x3072xbf16>
    %4303 = "ttir.typecast"(%4301, %4302) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4304 = ttir.empty() : tensor<1x640x3072xf32>
    %4305 = "ttir.typecast"(%4303, %4304) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4306 = ttir.empty() : tensor<1x640x3072xf32>
    %4307 = "ttir.multiply"(%4279, %4305, %4306) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4308 = ttir.empty() : tensor<1x640x3072xbf16>
    %4309 = "ttir.typecast"(%4307, %4308) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4310 = ttir.empty() : tensor<640x3072xbf16>
    %4311 = "ttir.reshape"(%4309, %4310) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4312 = ttir.empty() : tensor<1x8192x3072xbf16>
    %4313 = "ttir.reshape"(%arg142, %4312) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %4314 = ttir.empty() : tensor<8192x3072xbf16>
    %4315 = "ttir.reshape"(%4313, %4314) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4316 = ttir.empty() : tensor<3072x8192xbf16>
    %4317 = "ttir.permute"(%4315, %4316) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4318 = "ttir.dot_general"(%4311, %4317) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %4319 = ttir.empty() : tensor<1x640x8192xbf16>
    %4320 = "ttir.reshape"(%4318, %4319) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4321 = ttir.empty() : tensor<1x640x8192xf32>
    %4322 = "ttir.typecast"(%4320, %4321) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4323 = ttir.empty() : tensor<1x640x8192xbf16>
    %4324 = "ttir.sigmoid"(%4320, %4323) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4325 = ttir.empty() : tensor<1x640x8192xf32>
    %4326 = "ttir.typecast"(%4324, %4325) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4327 = ttir.empty() : tensor<1x640x8192xf32>
    %4328 = "ttir.multiply"(%4322, %4326, %4327) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4329 = ttir.empty() : tensor<1x640x8192xbf16>
    %4330 = "ttir.typecast"(%4328, %4329) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4331 = ttir.empty() : tensor<1x640x8192xf32>
    %4332 = "ttir.typecast"(%4330, %4331) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4333 = ttir.empty() : tensor<1x8192x3072xbf16>
    %4334 = "ttir.reshape"(%arg137, %4333) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %4335 = ttir.empty() : tensor<8192x3072xbf16>
    %4336 = "ttir.reshape"(%4334, %4335) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4337 = ttir.empty() : tensor<3072x8192xbf16>
    %4338 = "ttir.permute"(%4336, %4337) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4339 = "ttir.dot_general"(%4311, %4338) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %4340 = ttir.empty() : tensor<1x640x8192xbf16>
    %4341 = "ttir.reshape"(%4339, %4340) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4342 = ttir.empty() : tensor<1x640x8192xf32>
    %4343 = "ttir.typecast"(%4341, %4342) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4344 = ttir.empty() : tensor<1x640x8192xf32>
    %4345 = "ttir.multiply"(%4332, %4343, %4344) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4346 = ttir.empty() : tensor<1x640x8192xbf16>
    %4347 = "ttir.typecast"(%4345, %4346) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4348 = ttir.empty() : tensor<640x8192xbf16>
    %4349 = "ttir.reshape"(%4347, %4348) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %4350 = ttir.empty() : tensor<1x3072x8192xbf16>
    %4351 = "ttir.reshape"(%arg136, %4350) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %4352 = ttir.empty() : tensor<3072x8192xbf16>
    %4353 = "ttir.reshape"(%4351, %4352) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4354 = ttir.empty() : tensor<8192x3072xbf16>
    %4355 = "ttir.permute"(%4353, %4354) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4356 = "ttir.dot_general"(%4349, %4355) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %4357 = ttir.empty() : tensor<1x640x3072xbf16>
    %4358 = "ttir.reshape"(%4356, %4357) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4359 = ttir.empty() : tensor<1x640x3072xbf16>
    %4360 = "ttir.add"(%4269, %4358, %4359) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4361 = ttir.empty() : tensor<1x640x3072xf32>
    %4362 = "ttir.typecast"(%4360, %4361) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4363 = ttir.empty() : tensor<1x640x3072xf32>
    %4364 = "ttir.pow"(%4362, %5, %4363) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4365 = ttir.empty() : tensor<1x640xf32>
    %4366 = "ttir.sum"(%4364, %4365) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4367 = ttir.empty() : tensor<1x640xf32>
    %4368 = "ttir.multiply"(%4366, %4, %4367) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4369 = ttir.empty() : tensor<1x640x1xf32>
    %4370 = "ttir.reshape"(%4368, %4369) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4371 = ttir.empty() : tensor<1x640x1xf32>
    %4372 = "ttir.add"(%4370, %46, %4371) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4373 = ttir.empty() : tensor<1x640x1xf32>
    %4374 = "ttir.rsqrt"(%4372, %4373) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4375 = ttir.empty() : tensor<1x640xf32>
    %4376 = "ttir.reshape"(%4374, %4375) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4377 = ttir.empty() : tensor<1x640x1xf32>
    %4378 = "ttir.reshape"(%4376, %4377) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4379 = ttir.empty() : tensor<1x640x3072xf32>
    %4380 = "ttir.broadcast"(%4378, %4379) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4381 = ttir.empty() : tensor<1x640x3072xf32>
    %4382 = "ttir.multiply"(%4362, %4380, %4381) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4383 = ttir.empty() : tensor<1x640x3072xbf16>
    %4384 = "ttir.typecast"(%4382, %4383) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4385 = ttir.empty() : tensor<1x640x3072xf32>
    %4386 = "ttir.typecast"(%4384, %4385) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4387 = ttir.empty() : tensor<1x640x3072xf32>
    %4388 = "ttir.multiply"(%4132, %4386, %4387) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4389 = ttir.empty() : tensor<1x640x3072xbf16>
    %4390 = "ttir.typecast"(%4388, %4389) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4391 = ttir.empty() : tensor<640x3072xbf16>
    %4392 = "ttir.reshape"(%4390, %4391) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4393 = ttir.empty() : tensor<1x1024x3072xbf16>
    %4394 = "ttir.reshape"(%arg135, %4393) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %4395 = ttir.empty() : tensor<1024x3072xbf16>
    %4396 = "ttir.reshape"(%4394, %4395) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %4397 = ttir.empty() : tensor<3072x1024xbf16>
    %4398 = "ttir.permute"(%4396, %4397) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %4399 = "ttir.dot_general"(%4392, %4398) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %4400 = ttir.empty() : tensor<1x640x8x128xbf16>
    %4401 = "ttir.reshape"(%4399, %4400) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %4402 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4403 = "ttir.permute"(%4401, %4402) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4404 = ttir.empty() : tensor<1x1x3072xbf16>
    %4405 = "ttir.reshape"(%arg152, %4404) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %4406 = ttir.empty() : tensor<3072xbf16>
    %4407 = "ttir.reshape"(%4405, %4406) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %4408 = ttir.empty() : tensor<3072xf32>
    %4409 = "ttir.typecast"(%4407, %4408) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %4410 = ttir.empty() : tensor<1x1x3072xf32>
    %4411 = "ttir.reshape"(%4409, %4410) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %4412 = ttir.empty() : tensor<1x640x3072xf32>
    %4413 = "ttir.broadcast"(%4411, %4412) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4414 = ttir.empty() : tensor<1x3072x3072xbf16>
    %4415 = "ttir.reshape"(%arg149, %4414) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %4416 = ttir.empty() : tensor<3072x3072xbf16>
    %4417 = "ttir.reshape"(%4415, %4416) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4418 = ttir.empty() : tensor<3072x3072xbf16>
    %4419 = "ttir.permute"(%4417, %4418) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4420 = "ttir.dot_general"(%4392, %4419) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %4421 = ttir.empty() : tensor<1x640x24x128xbf16>
    %4422 = "ttir.reshape"(%4420, %4421) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %4423 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4424 = "ttir.permute"(%4422, %4423) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4425 = ttir.empty() : tensor<1x24x640x128xf32>
    %4426 = "ttir.typecast"(%4424, %4425) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4427 = ttir.empty() : tensor<1x24x640x128xf32>
    %4428 = "ttir.multiply"(%4426, %125, %4427) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4429 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4430 = "ttir.typecast"(%4428, %4429) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4431 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4432 = "ttir.slice_static"(%4424, %4431) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4433 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4434 = "ttir.neg"(%4432, %4433) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4435 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4436 = "ttir.slice_static"(%4424, %4435) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4437 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4438 = "ttir.concat"(%4434, %4436, %4437) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4439 = ttir.empty() : tensor<1x24x640x128xf32>
    %4440 = "ttir.typecast"(%4438, %4439) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4441 = ttir.empty() : tensor<1x24x640x128xf32>
    %4442 = "ttir.multiply"(%4440, %153, %4441) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4443 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4444 = "ttir.typecast"(%4442, %4443) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4445 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4446 = "ttir.add"(%4430, %4444, %4445) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4447 = ttir.empty() : tensor<24x640x128xbf16>
    %4448 = "ttir.reshape"(%4446, %4447) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %4449 = ttir.empty() : tensor<1x1024x3072xbf16>
    %4450 = "ttir.reshape"(%arg148, %4449) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %4451 = ttir.empty() : tensor<1024x3072xbf16>
    %4452 = "ttir.reshape"(%4450, %4451) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %4453 = ttir.empty() : tensor<3072x1024xbf16>
    %4454 = "ttir.permute"(%4452, %4453) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %4455 = "ttir.dot_general"(%4392, %4454) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %4456 = ttir.empty() : tensor<1x640x8x128xbf16>
    %4457 = "ttir.reshape"(%4455, %4456) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %4458 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4459 = "ttir.permute"(%4457, %4458) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4460 = ttir.empty() : tensor<1x8x640x128xf32>
    %4461 = "ttir.typecast"(%4459, %4460) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4462 = ttir.empty() : tensor<1x8x640x128xf32>
    %4463 = "ttir.multiply"(%4461, %178, %4462) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4464 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4465 = "ttir.typecast"(%4463, %4464) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4466 = ttir.empty() : tensor<1x8x640x64xbf16>
    %4467 = "ttir.slice_static"(%4459, %4466) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %4468 = ttir.empty() : tensor<1x8x640x64xbf16>
    %4469 = "ttir.neg"(%4467, %4468) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %4470 = ttir.empty() : tensor<1x8x640x64xbf16>
    %4471 = "ttir.slice_static"(%4459, %4470) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %4472 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4473 = "ttir.concat"(%4469, %4471, %4472) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4474 = ttir.empty() : tensor<1x8x640x128xf32>
    %4475 = "ttir.typecast"(%4473, %4474) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4476 = ttir.empty() : tensor<1x8x640x128xf32>
    %4477 = "ttir.multiply"(%4475, %196, %4476) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4478 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4479 = "ttir.typecast"(%4477, %4478) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4480 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4481 = "ttir.add"(%4465, %4479, %4480) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4482 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %4483 = "ttir.reshape"(%4481, %4482) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %4484 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %4485 = "ttir.broadcast"(%4483, %4484) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %4486 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4487 = "ttir.reshape"(%4485, %4486) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4488 = ttir.empty() : tensor<1x24x128x640xbf16>
    %4489 = "ttir.permute"(%4487, %4488) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %4490 = ttir.empty() : tensor<24x128x640xbf16>
    %4491 = "ttir.reshape"(%4489, %4490) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %4492 = "ttir.dot_general"(%4448, %4491) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %4493 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4494 = "ttir.reshape"(%4492, %4493) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4495 = ttir.empty() : tensor<1x24x640x640xf32>
    %4496 = "ttir.typecast"(%4494, %4495) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4497 = ttir.empty() : tensor<1x24x640x640xf32>
    %4498 = "ttir.multiply"(%4496, %221, %4497) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4499 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4500 = "ttir.typecast"(%4498, %4499) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4501 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4502 = "ttir.add"(%4500, %285, %4501) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4503 = ttir.empty() : tensor<1x24x640x640xf32>
    %4504 = "ttir.typecast"(%4502, %4503) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4505 = ttir.empty() : tensor<1x24x640xf32>
    %4506 = "ttir.max"(%4504, %4505) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %4507 = ttir.empty() : tensor<1x24x640x1xf32>
    %4508 = "ttir.reshape"(%4506, %4507) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %4509 = ttir.empty() : tensor<1x24x640x640xf32>
    %4510 = "ttir.broadcast"(%4508, %4509) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4511 = ttir.empty() : tensor<1x24x640x640xf32>
    %4512 = "ttir.subtract"(%4504, %4510, %4511) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4513 = ttir.empty() : tensor<1x24x640x640xf32>
    %4514 = "ttir.exp"(%4512, %4513) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4515 = ttir.empty() : tensor<1x24x640xf32>
    %4516 = "ttir.sum"(%4514, %4515) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %4517 = ttir.empty() : tensor<1x24x640x1xf32>
    %4518 = "ttir.reshape"(%4516, %4517) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %4519 = ttir.empty() : tensor<1x24x640x640xf32>
    %4520 = "ttir.broadcast"(%4518, %4519) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4521 = ttir.empty() : tensor<1x24x640x640xf32>
    %4522 = "ttir.div"(%4514, %4520, %4521) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4523 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4524 = "ttir.typecast"(%4522, %4523) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4525 = ttir.empty() : tensor<24x640x640xbf16>
    %4526 = "ttir.reshape"(%4524, %4525) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %4527 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %4528 = "ttir.reshape"(%4403, %4527) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %4529 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %4530 = "ttir.broadcast"(%4528, %4529) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %4531 = ttir.empty() : tensor<24x640x128xbf16>
    %4532 = "ttir.reshape"(%4530, %4531) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %4533 = "ttir.dot_general"(%4526, %4532) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %4534 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4535 = "ttir.reshape"(%4533, %4534) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4536 = ttir.empty() : tensor<1x640x24x128xbf16>
    %4537 = "ttir.permute"(%4535, %4536) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %4538 = ttir.empty() : tensor<640x3072xbf16>
    %4539 = "ttir.reshape"(%4537, %4538) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4540 = ttir.empty() : tensor<1x3072x3072xbf16>
    %4541 = "ttir.reshape"(%arg147, %4540) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %4542 = ttir.empty() : tensor<3072x3072xbf16>
    %4543 = "ttir.reshape"(%4541, %4542) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4544 = ttir.empty() : tensor<3072x3072xbf16>
    %4545 = "ttir.permute"(%4543, %4544) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4546 = "ttir.dot_general"(%4539, %4545) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %4547 = ttir.empty() : tensor<1x640x3072xbf16>
    %4548 = "ttir.reshape"(%4546, %4547) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4549 = ttir.empty() : tensor<1x640x3072xbf16>
    %4550 = "ttir.add"(%4360, %4548, %4549) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4551 = ttir.empty() : tensor<1x1x3072xbf16>
    %4552 = "ttir.reshape"(%arg150, %4551) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %4553 = ttir.empty() : tensor<3072xbf16>
    %4554 = "ttir.reshape"(%4552, %4553) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %4555 = ttir.empty() : tensor<3072xf32>
    %4556 = "ttir.typecast"(%4554, %4555) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %4557 = ttir.empty() : tensor<1x1x3072xf32>
    %4558 = "ttir.reshape"(%4556, %4557) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %4559 = ttir.empty() : tensor<1x640x3072xf32>
    %4560 = "ttir.broadcast"(%4558, %4559) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4561 = ttir.empty() : tensor<1x640x3072xf32>
    %4562 = "ttir.typecast"(%4550, %4561) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4563 = ttir.empty() : tensor<1x640x3072xf32>
    %4564 = "ttir.pow"(%4562, %5, %4563) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4565 = ttir.empty() : tensor<1x640xf32>
    %4566 = "ttir.sum"(%4564, %4565) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4567 = ttir.empty() : tensor<1x640xf32>
    %4568 = "ttir.multiply"(%4566, %4, %4567) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4569 = ttir.empty() : tensor<1x640x1xf32>
    %4570 = "ttir.reshape"(%4568, %4569) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4571 = ttir.empty() : tensor<1x640x1xf32>
    %4572 = "ttir.add"(%4570, %46, %4571) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4573 = ttir.empty() : tensor<1x640x1xf32>
    %4574 = "ttir.rsqrt"(%4572, %4573) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4575 = ttir.empty() : tensor<1x640xf32>
    %4576 = "ttir.reshape"(%4574, %4575) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4577 = ttir.empty() : tensor<1x640x1xf32>
    %4578 = "ttir.reshape"(%4576, %4577) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4579 = ttir.empty() : tensor<1x640x3072xf32>
    %4580 = "ttir.broadcast"(%4578, %4579) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4581 = ttir.empty() : tensor<1x640x3072xf32>
    %4582 = "ttir.multiply"(%4562, %4580, %4581) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4583 = ttir.empty() : tensor<1x640x3072xbf16>
    %4584 = "ttir.typecast"(%4582, %4583) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4585 = ttir.empty() : tensor<1x640x3072xf32>
    %4586 = "ttir.typecast"(%4584, %4585) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4587 = ttir.empty() : tensor<1x640x3072xf32>
    %4588 = "ttir.multiply"(%4560, %4586, %4587) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4589 = ttir.empty() : tensor<1x640x3072xbf16>
    %4590 = "ttir.typecast"(%4588, %4589) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4591 = ttir.empty() : tensor<640x3072xbf16>
    %4592 = "ttir.reshape"(%4590, %4591) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4593 = ttir.empty() : tensor<1x8192x3072xbf16>
    %4594 = "ttir.reshape"(%arg151, %4593) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %4595 = ttir.empty() : tensor<8192x3072xbf16>
    %4596 = "ttir.reshape"(%4594, %4595) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4597 = ttir.empty() : tensor<3072x8192xbf16>
    %4598 = "ttir.permute"(%4596, %4597) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4599 = "ttir.dot_general"(%4592, %4598) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %4600 = ttir.empty() : tensor<1x640x8192xbf16>
    %4601 = "ttir.reshape"(%4599, %4600) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4602 = ttir.empty() : tensor<1x640x8192xf32>
    %4603 = "ttir.typecast"(%4601, %4602) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4604 = ttir.empty() : tensor<1x640x8192xbf16>
    %4605 = "ttir.sigmoid"(%4601, %4604) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4606 = ttir.empty() : tensor<1x640x8192xf32>
    %4607 = "ttir.typecast"(%4605, %4606) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4608 = ttir.empty() : tensor<1x640x8192xf32>
    %4609 = "ttir.multiply"(%4603, %4607, %4608) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4610 = ttir.empty() : tensor<1x640x8192xbf16>
    %4611 = "ttir.typecast"(%4609, %4610) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4612 = ttir.empty() : tensor<1x640x8192xf32>
    %4613 = "ttir.typecast"(%4611, %4612) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4614 = ttir.empty() : tensor<1x8192x3072xbf16>
    %4615 = "ttir.reshape"(%arg146, %4614) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %4616 = ttir.empty() : tensor<8192x3072xbf16>
    %4617 = "ttir.reshape"(%4615, %4616) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4618 = ttir.empty() : tensor<3072x8192xbf16>
    %4619 = "ttir.permute"(%4617, %4618) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4620 = "ttir.dot_general"(%4592, %4619) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %4621 = ttir.empty() : tensor<1x640x8192xbf16>
    %4622 = "ttir.reshape"(%4620, %4621) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4623 = ttir.empty() : tensor<1x640x8192xf32>
    %4624 = "ttir.typecast"(%4622, %4623) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4625 = ttir.empty() : tensor<1x640x8192xf32>
    %4626 = "ttir.multiply"(%4613, %4624, %4625) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4627 = ttir.empty() : tensor<1x640x8192xbf16>
    %4628 = "ttir.typecast"(%4626, %4627) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4629 = ttir.empty() : tensor<640x8192xbf16>
    %4630 = "ttir.reshape"(%4628, %4629) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %4631 = ttir.empty() : tensor<1x3072x8192xbf16>
    %4632 = "ttir.reshape"(%arg145, %4631) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %4633 = ttir.empty() : tensor<3072x8192xbf16>
    %4634 = "ttir.reshape"(%4632, %4633) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4635 = ttir.empty() : tensor<8192x3072xbf16>
    %4636 = "ttir.permute"(%4634, %4635) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4637 = "ttir.dot_general"(%4630, %4636) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %4638 = ttir.empty() : tensor<1x640x3072xbf16>
    %4639 = "ttir.reshape"(%4637, %4638) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4640 = ttir.empty() : tensor<1x640x3072xbf16>
    %4641 = "ttir.add"(%4550, %4639, %4640) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4642 = ttir.empty() : tensor<1x640x3072xf32>
    %4643 = "ttir.typecast"(%4641, %4642) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4644 = ttir.empty() : tensor<1x640x3072xf32>
    %4645 = "ttir.pow"(%4643, %5, %4644) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4646 = ttir.empty() : tensor<1x640xf32>
    %4647 = "ttir.sum"(%4645, %4646) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4648 = ttir.empty() : tensor<1x640xf32>
    %4649 = "ttir.multiply"(%4647, %4, %4648) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4650 = ttir.empty() : tensor<1x640x1xf32>
    %4651 = "ttir.reshape"(%4649, %4650) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4652 = ttir.empty() : tensor<1x640x1xf32>
    %4653 = "ttir.add"(%4651, %46, %4652) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4654 = ttir.empty() : tensor<1x640x1xf32>
    %4655 = "ttir.rsqrt"(%4653, %4654) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4656 = ttir.empty() : tensor<1x640xf32>
    %4657 = "ttir.reshape"(%4655, %4656) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4658 = ttir.empty() : tensor<1x640x1xf32>
    %4659 = "ttir.reshape"(%4657, %4658) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4660 = ttir.empty() : tensor<1x640x3072xf32>
    %4661 = "ttir.broadcast"(%4659, %4660) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4662 = ttir.empty() : tensor<1x640x3072xf32>
    %4663 = "ttir.multiply"(%4643, %4661, %4662) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4664 = ttir.empty() : tensor<1x640x3072xbf16>
    %4665 = "ttir.typecast"(%4663, %4664) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4666 = ttir.empty() : tensor<1x640x3072xf32>
    %4667 = "ttir.typecast"(%4665, %4666) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4668 = ttir.empty() : tensor<1x640x3072xf32>
    %4669 = "ttir.multiply"(%4413, %4667, %4668) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4670 = ttir.empty() : tensor<1x640x3072xbf16>
    %4671 = "ttir.typecast"(%4669, %4670) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4672 = ttir.empty() : tensor<640x3072xbf16>
    %4673 = "ttir.reshape"(%4671, %4672) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4674 = ttir.empty() : tensor<1x1024x3072xbf16>
    %4675 = "ttir.reshape"(%arg144, %4674) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %4676 = ttir.empty() : tensor<1024x3072xbf16>
    %4677 = "ttir.reshape"(%4675, %4676) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %4678 = ttir.empty() : tensor<3072x1024xbf16>
    %4679 = "ttir.permute"(%4677, %4678) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %4680 = "ttir.dot_general"(%4673, %4679) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %4681 = ttir.empty() : tensor<1x640x8x128xbf16>
    %4682 = "ttir.reshape"(%4680, %4681) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %4683 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4684 = "ttir.permute"(%4682, %4683) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4685 = ttir.empty() : tensor<1x1x3072xbf16>
    %4686 = "ttir.reshape"(%arg161, %4685) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %4687 = ttir.empty() : tensor<3072xbf16>
    %4688 = "ttir.reshape"(%4686, %4687) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %4689 = ttir.empty() : tensor<3072xf32>
    %4690 = "ttir.typecast"(%4688, %4689) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %4691 = ttir.empty() : tensor<1x1x3072xf32>
    %4692 = "ttir.reshape"(%4690, %4691) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %4693 = ttir.empty() : tensor<1x640x3072xf32>
    %4694 = "ttir.broadcast"(%4692, %4693) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4695 = ttir.empty() : tensor<1x3072x3072xbf16>
    %4696 = "ttir.reshape"(%arg158, %4695) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %4697 = ttir.empty() : tensor<3072x3072xbf16>
    %4698 = "ttir.reshape"(%4696, %4697) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4699 = ttir.empty() : tensor<3072x3072xbf16>
    %4700 = "ttir.permute"(%4698, %4699) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4701 = "ttir.dot_general"(%4673, %4700) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %4702 = ttir.empty() : tensor<1x640x24x128xbf16>
    %4703 = "ttir.reshape"(%4701, %4702) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %4704 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4705 = "ttir.permute"(%4703, %4704) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4706 = ttir.empty() : tensor<1x24x640x128xf32>
    %4707 = "ttir.typecast"(%4705, %4706) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4708 = ttir.empty() : tensor<1x24x640x128xf32>
    %4709 = "ttir.multiply"(%4707, %125, %4708) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4710 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4711 = "ttir.typecast"(%4709, %4710) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4712 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4713 = "ttir.slice_static"(%4705, %4712) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4714 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4715 = "ttir.neg"(%4713, %4714) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4716 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4717 = "ttir.slice_static"(%4705, %4716) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4718 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4719 = "ttir.concat"(%4715, %4717, %4718) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4720 = ttir.empty() : tensor<1x24x640x128xf32>
    %4721 = "ttir.typecast"(%4719, %4720) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4722 = ttir.empty() : tensor<1x24x640x128xf32>
    %4723 = "ttir.multiply"(%4721, %153, %4722) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4724 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4725 = "ttir.typecast"(%4723, %4724) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4726 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4727 = "ttir.add"(%4711, %4725, %4726) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4728 = ttir.empty() : tensor<24x640x128xbf16>
    %4729 = "ttir.reshape"(%4727, %4728) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %4730 = ttir.empty() : tensor<1x1024x3072xbf16>
    %4731 = "ttir.reshape"(%arg157, %4730) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %4732 = ttir.empty() : tensor<1024x3072xbf16>
    %4733 = "ttir.reshape"(%4731, %4732) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %4734 = ttir.empty() : tensor<3072x1024xbf16>
    %4735 = "ttir.permute"(%4733, %4734) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %4736 = "ttir.dot_general"(%4673, %4735) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %4737 = ttir.empty() : tensor<1x640x8x128xbf16>
    %4738 = "ttir.reshape"(%4736, %4737) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %4739 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4740 = "ttir.permute"(%4738, %4739) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4741 = ttir.empty() : tensor<1x8x640x128xf32>
    %4742 = "ttir.typecast"(%4740, %4741) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4743 = ttir.empty() : tensor<1x8x640x128xf32>
    %4744 = "ttir.multiply"(%4742, %178, %4743) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4745 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4746 = "ttir.typecast"(%4744, %4745) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4747 = ttir.empty() : tensor<1x8x640x64xbf16>
    %4748 = "ttir.slice_static"(%4740, %4747) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %4749 = ttir.empty() : tensor<1x8x640x64xbf16>
    %4750 = "ttir.neg"(%4748, %4749) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %4751 = ttir.empty() : tensor<1x8x640x64xbf16>
    %4752 = "ttir.slice_static"(%4740, %4751) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %4753 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4754 = "ttir.concat"(%4750, %4752, %4753) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4755 = ttir.empty() : tensor<1x8x640x128xf32>
    %4756 = "ttir.typecast"(%4754, %4755) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4757 = ttir.empty() : tensor<1x8x640x128xf32>
    %4758 = "ttir.multiply"(%4756, %196, %4757) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %4759 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4760 = "ttir.typecast"(%4758, %4759) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4761 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4762 = "ttir.add"(%4746, %4760, %4761) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4763 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %4764 = "ttir.reshape"(%4762, %4763) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %4765 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %4766 = "ttir.broadcast"(%4764, %4765) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %4767 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4768 = "ttir.reshape"(%4766, %4767) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4769 = ttir.empty() : tensor<1x24x128x640xbf16>
    %4770 = "ttir.permute"(%4768, %4769) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %4771 = ttir.empty() : tensor<24x128x640xbf16>
    %4772 = "ttir.reshape"(%4770, %4771) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %4773 = "ttir.dot_general"(%4729, %4772) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %4774 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4775 = "ttir.reshape"(%4773, %4774) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4776 = ttir.empty() : tensor<1x24x640x640xf32>
    %4777 = "ttir.typecast"(%4775, %4776) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4778 = ttir.empty() : tensor<1x24x640x640xf32>
    %4779 = "ttir.multiply"(%4777, %221, %4778) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4780 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4781 = "ttir.typecast"(%4779, %4780) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4782 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4783 = "ttir.add"(%4781, %285, %4782) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4784 = ttir.empty() : tensor<1x24x640x640xf32>
    %4785 = "ttir.typecast"(%4783, %4784) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4786 = ttir.empty() : tensor<1x24x640xf32>
    %4787 = "ttir.max"(%4785, %4786) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %4788 = ttir.empty() : tensor<1x24x640x1xf32>
    %4789 = "ttir.reshape"(%4787, %4788) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %4790 = ttir.empty() : tensor<1x24x640x640xf32>
    %4791 = "ttir.broadcast"(%4789, %4790) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4792 = ttir.empty() : tensor<1x24x640x640xf32>
    %4793 = "ttir.subtract"(%4785, %4791, %4792) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4794 = ttir.empty() : tensor<1x24x640x640xf32>
    %4795 = "ttir.exp"(%4793, %4794) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4796 = ttir.empty() : tensor<1x24x640xf32>
    %4797 = "ttir.sum"(%4795, %4796) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %4798 = ttir.empty() : tensor<1x24x640x1xf32>
    %4799 = "ttir.reshape"(%4797, %4798) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %4800 = ttir.empty() : tensor<1x24x640x640xf32>
    %4801 = "ttir.broadcast"(%4799, %4800) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4802 = ttir.empty() : tensor<1x24x640x640xf32>
    %4803 = "ttir.div"(%4795, %4801, %4802) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %4804 = ttir.empty() : tensor<1x24x640x640xbf16>
    %4805 = "ttir.typecast"(%4803, %4804) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %4806 = ttir.empty() : tensor<24x640x640xbf16>
    %4807 = "ttir.reshape"(%4805, %4806) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %4808 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %4809 = "ttir.reshape"(%4684, %4808) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %4810 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %4811 = "ttir.broadcast"(%4809, %4810) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %4812 = ttir.empty() : tensor<24x640x128xbf16>
    %4813 = "ttir.reshape"(%4811, %4812) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %4814 = "ttir.dot_general"(%4807, %4813) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %4815 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4816 = "ttir.reshape"(%4814, %4815) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4817 = ttir.empty() : tensor<1x640x24x128xbf16>
    %4818 = "ttir.permute"(%4816, %4817) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %4819 = ttir.empty() : tensor<640x3072xbf16>
    %4820 = "ttir.reshape"(%4818, %4819) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4821 = ttir.empty() : tensor<1x3072x3072xbf16>
    %4822 = "ttir.reshape"(%arg156, %4821) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %4823 = ttir.empty() : tensor<3072x3072xbf16>
    %4824 = "ttir.reshape"(%4822, %4823) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4825 = ttir.empty() : tensor<3072x3072xbf16>
    %4826 = "ttir.permute"(%4824, %4825) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4827 = "ttir.dot_general"(%4820, %4826) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %4828 = ttir.empty() : tensor<1x640x3072xbf16>
    %4829 = "ttir.reshape"(%4827, %4828) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4830 = ttir.empty() : tensor<1x640x3072xbf16>
    %4831 = "ttir.add"(%4641, %4829, %4830) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4832 = ttir.empty() : tensor<1x1x3072xbf16>
    %4833 = "ttir.reshape"(%arg159, %4832) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %4834 = ttir.empty() : tensor<3072xbf16>
    %4835 = "ttir.reshape"(%4833, %4834) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %4836 = ttir.empty() : tensor<3072xf32>
    %4837 = "ttir.typecast"(%4835, %4836) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %4838 = ttir.empty() : tensor<1x1x3072xf32>
    %4839 = "ttir.reshape"(%4837, %4838) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %4840 = ttir.empty() : tensor<1x640x3072xf32>
    %4841 = "ttir.broadcast"(%4839, %4840) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4842 = ttir.empty() : tensor<1x640x3072xf32>
    %4843 = "ttir.typecast"(%4831, %4842) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4844 = ttir.empty() : tensor<1x640x3072xf32>
    %4845 = "ttir.pow"(%4843, %5, %4844) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4846 = ttir.empty() : tensor<1x640xf32>
    %4847 = "ttir.sum"(%4845, %4846) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4848 = ttir.empty() : tensor<1x640xf32>
    %4849 = "ttir.multiply"(%4847, %4, %4848) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4850 = ttir.empty() : tensor<1x640x1xf32>
    %4851 = "ttir.reshape"(%4849, %4850) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4852 = ttir.empty() : tensor<1x640x1xf32>
    %4853 = "ttir.add"(%4851, %46, %4852) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4854 = ttir.empty() : tensor<1x640x1xf32>
    %4855 = "ttir.rsqrt"(%4853, %4854) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4856 = ttir.empty() : tensor<1x640xf32>
    %4857 = "ttir.reshape"(%4855, %4856) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4858 = ttir.empty() : tensor<1x640x1xf32>
    %4859 = "ttir.reshape"(%4857, %4858) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4860 = ttir.empty() : tensor<1x640x3072xf32>
    %4861 = "ttir.broadcast"(%4859, %4860) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4862 = ttir.empty() : tensor<1x640x3072xf32>
    %4863 = "ttir.multiply"(%4843, %4861, %4862) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4864 = ttir.empty() : tensor<1x640x3072xbf16>
    %4865 = "ttir.typecast"(%4863, %4864) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4866 = ttir.empty() : tensor<1x640x3072xf32>
    %4867 = "ttir.typecast"(%4865, %4866) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4868 = ttir.empty() : tensor<1x640x3072xf32>
    %4869 = "ttir.multiply"(%4841, %4867, %4868) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4870 = ttir.empty() : tensor<1x640x3072xbf16>
    %4871 = "ttir.typecast"(%4869, %4870) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4872 = ttir.empty() : tensor<640x3072xbf16>
    %4873 = "ttir.reshape"(%4871, %4872) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4874 = ttir.empty() : tensor<1x8192x3072xbf16>
    %4875 = "ttir.reshape"(%arg160, %4874) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %4876 = ttir.empty() : tensor<8192x3072xbf16>
    %4877 = "ttir.reshape"(%4875, %4876) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4878 = ttir.empty() : tensor<3072x8192xbf16>
    %4879 = "ttir.permute"(%4877, %4878) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4880 = "ttir.dot_general"(%4873, %4879) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %4881 = ttir.empty() : tensor<1x640x8192xbf16>
    %4882 = "ttir.reshape"(%4880, %4881) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4883 = ttir.empty() : tensor<1x640x8192xf32>
    %4884 = "ttir.typecast"(%4882, %4883) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4885 = ttir.empty() : tensor<1x640x8192xbf16>
    %4886 = "ttir.sigmoid"(%4882, %4885) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4887 = ttir.empty() : tensor<1x640x8192xf32>
    %4888 = "ttir.typecast"(%4886, %4887) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4889 = ttir.empty() : tensor<1x640x8192xf32>
    %4890 = "ttir.multiply"(%4884, %4888, %4889) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4891 = ttir.empty() : tensor<1x640x8192xbf16>
    %4892 = "ttir.typecast"(%4890, %4891) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4893 = ttir.empty() : tensor<1x640x8192xf32>
    %4894 = "ttir.typecast"(%4892, %4893) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4895 = ttir.empty() : tensor<1x8192x3072xbf16>
    %4896 = "ttir.reshape"(%arg155, %4895) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %4897 = ttir.empty() : tensor<8192x3072xbf16>
    %4898 = "ttir.reshape"(%4896, %4897) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4899 = ttir.empty() : tensor<3072x8192xbf16>
    %4900 = "ttir.permute"(%4898, %4899) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4901 = "ttir.dot_general"(%4873, %4900) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %4902 = ttir.empty() : tensor<1x640x8192xbf16>
    %4903 = "ttir.reshape"(%4901, %4902) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4904 = ttir.empty() : tensor<1x640x8192xf32>
    %4905 = "ttir.typecast"(%4903, %4904) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4906 = ttir.empty() : tensor<1x640x8192xf32>
    %4907 = "ttir.multiply"(%4894, %4905, %4906) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %4908 = ttir.empty() : tensor<1x640x8192xbf16>
    %4909 = "ttir.typecast"(%4907, %4908) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %4910 = ttir.empty() : tensor<640x8192xbf16>
    %4911 = "ttir.reshape"(%4909, %4910) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %4912 = ttir.empty() : tensor<1x3072x8192xbf16>
    %4913 = "ttir.reshape"(%arg154, %4912) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %4914 = ttir.empty() : tensor<3072x8192xbf16>
    %4915 = "ttir.reshape"(%4913, %4914) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %4916 = ttir.empty() : tensor<8192x3072xbf16>
    %4917 = "ttir.permute"(%4915, %4916) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %4918 = "ttir.dot_general"(%4911, %4917) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %4919 = ttir.empty() : tensor<1x640x3072xbf16>
    %4920 = "ttir.reshape"(%4918, %4919) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4921 = ttir.empty() : tensor<1x640x3072xbf16>
    %4922 = "ttir.add"(%4831, %4920, %4921) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4923 = ttir.empty() : tensor<1x640x3072xf32>
    %4924 = "ttir.typecast"(%4922, %4923) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4925 = ttir.empty() : tensor<1x640x3072xf32>
    %4926 = "ttir.pow"(%4924, %5, %4925) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4927 = ttir.empty() : tensor<1x640xf32>
    %4928 = "ttir.sum"(%4926, %4927) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4929 = ttir.empty() : tensor<1x640xf32>
    %4930 = "ttir.multiply"(%4928, %4, %4929) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4931 = ttir.empty() : tensor<1x640x1xf32>
    %4932 = "ttir.reshape"(%4930, %4931) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4933 = ttir.empty() : tensor<1x640x1xf32>
    %4934 = "ttir.add"(%4932, %46, %4933) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4935 = ttir.empty() : tensor<1x640x1xf32>
    %4936 = "ttir.rsqrt"(%4934, %4935) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4937 = ttir.empty() : tensor<1x640xf32>
    %4938 = "ttir.reshape"(%4936, %4937) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4939 = ttir.empty() : tensor<1x640x1xf32>
    %4940 = "ttir.reshape"(%4938, %4939) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %4941 = ttir.empty() : tensor<1x640x3072xf32>
    %4942 = "ttir.broadcast"(%4940, %4941) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4943 = ttir.empty() : tensor<1x640x3072xf32>
    %4944 = "ttir.multiply"(%4924, %4942, %4943) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4945 = ttir.empty() : tensor<1x640x3072xbf16>
    %4946 = "ttir.typecast"(%4944, %4945) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4947 = ttir.empty() : tensor<1x640x3072xf32>
    %4948 = "ttir.typecast"(%4946, %4947) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4949 = ttir.empty() : tensor<1x640x3072xf32>
    %4950 = "ttir.multiply"(%4694, %4948, %4949) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4951 = ttir.empty() : tensor<1x640x3072xbf16>
    %4952 = "ttir.typecast"(%4950, %4951) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %4953 = ttir.empty() : tensor<640x3072xbf16>
    %4954 = "ttir.reshape"(%4952, %4953) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %4955 = ttir.empty() : tensor<1x1024x3072xbf16>
    %4956 = "ttir.reshape"(%arg153, %4955) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %4957 = ttir.empty() : tensor<1024x3072xbf16>
    %4958 = "ttir.reshape"(%4956, %4957) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %4959 = ttir.empty() : tensor<3072x1024xbf16>
    %4960 = "ttir.permute"(%4958, %4959) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %4961 = "ttir.dot_general"(%4954, %4960) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %4962 = ttir.empty() : tensor<1x640x8x128xbf16>
    %4963 = "ttir.reshape"(%4961, %4962) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %4964 = ttir.empty() : tensor<1x8x640x128xbf16>
    %4965 = "ttir.permute"(%4963, %4964) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %4966 = ttir.empty() : tensor<1x1x3072xbf16>
    %4967 = "ttir.reshape"(%arg170, %4966) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %4968 = ttir.empty() : tensor<3072xbf16>
    %4969 = "ttir.reshape"(%4967, %4968) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %4970 = ttir.empty() : tensor<3072xf32>
    %4971 = "ttir.typecast"(%4969, %4970) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %4972 = ttir.empty() : tensor<1x1x3072xf32>
    %4973 = "ttir.reshape"(%4971, %4972) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %4974 = ttir.empty() : tensor<1x640x3072xf32>
    %4975 = "ttir.broadcast"(%4973, %4974) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %4976 = ttir.empty() : tensor<1x3072x3072xbf16>
    %4977 = "ttir.reshape"(%arg167, %4976) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %4978 = ttir.empty() : tensor<3072x3072xbf16>
    %4979 = "ttir.reshape"(%4977, %4978) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4980 = ttir.empty() : tensor<3072x3072xbf16>
    %4981 = "ttir.permute"(%4979, %4980) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4982 = "ttir.dot_general"(%4954, %4981) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %4983 = ttir.empty() : tensor<1x640x24x128xbf16>
    %4984 = "ttir.reshape"(%4982, %4983) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %4985 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4986 = "ttir.permute"(%4984, %4985) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4987 = ttir.empty() : tensor<1x24x640x128xf32>
    %4988 = "ttir.typecast"(%4986, %4987) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4989 = ttir.empty() : tensor<1x24x640x128xf32>
    %4990 = "ttir.multiply"(%4988, %125, %4989) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %4991 = ttir.empty() : tensor<1x24x640x128xbf16>
    %4992 = "ttir.typecast"(%4990, %4991) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %4993 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4994 = "ttir.slice_static"(%4986, %4993) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4995 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4996 = "ttir.neg"(%4994, %4995) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4997 = ttir.empty() : tensor<1x24x640x64xbf16>
    %4998 = "ttir.slice_static"(%4986, %4997) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %4999 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5000 = "ttir.concat"(%4996, %4998, %4999) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5001 = ttir.empty() : tensor<1x24x640x128xf32>
    %5002 = "ttir.typecast"(%5000, %5001) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5003 = ttir.empty() : tensor<1x24x640x128xf32>
    %5004 = "ttir.multiply"(%5002, %153, %5003) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5005 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5006 = "ttir.typecast"(%5004, %5005) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5007 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5008 = "ttir.add"(%4992, %5006, %5007) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5009 = ttir.empty() : tensor<24x640x128xbf16>
    %5010 = "ttir.reshape"(%5008, %5009) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5011 = ttir.empty() : tensor<1x1024x3072xbf16>
    %5012 = "ttir.reshape"(%arg166, %5011) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %5013 = ttir.empty() : tensor<1024x3072xbf16>
    %5014 = "ttir.reshape"(%5012, %5013) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %5015 = ttir.empty() : tensor<3072x1024xbf16>
    %5016 = "ttir.permute"(%5014, %5015) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %5017 = "ttir.dot_general"(%4954, %5016) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %5018 = ttir.empty() : tensor<1x640x8x128xbf16>
    %5019 = "ttir.reshape"(%5017, %5018) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %5020 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5021 = "ttir.permute"(%5019, %5020) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5022 = ttir.empty() : tensor<1x8x640x128xf32>
    %5023 = "ttir.typecast"(%5021, %5022) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5024 = ttir.empty() : tensor<1x8x640x128xf32>
    %5025 = "ttir.multiply"(%5023, %178, %5024) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5026 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5027 = "ttir.typecast"(%5025, %5026) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5028 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5029 = "ttir.slice_static"(%5021, %5028) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5030 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5031 = "ttir.neg"(%5029, %5030) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5032 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5033 = "ttir.slice_static"(%5021, %5032) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5034 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5035 = "ttir.concat"(%5031, %5033, %5034) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5036 = ttir.empty() : tensor<1x8x640x128xf32>
    %5037 = "ttir.typecast"(%5035, %5036) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5038 = ttir.empty() : tensor<1x8x640x128xf32>
    %5039 = "ttir.multiply"(%5037, %196, %5038) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5040 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5041 = "ttir.typecast"(%5039, %5040) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5042 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5043 = "ttir.add"(%5027, %5041, %5042) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5044 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %5045 = "ttir.reshape"(%5043, %5044) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %5046 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %5047 = "ttir.broadcast"(%5045, %5046) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %5048 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5049 = "ttir.reshape"(%5047, %5048) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5050 = ttir.empty() : tensor<1x24x128x640xbf16>
    %5051 = "ttir.permute"(%5049, %5050) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %5052 = ttir.empty() : tensor<24x128x640xbf16>
    %5053 = "ttir.reshape"(%5051, %5052) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %5054 = "ttir.dot_general"(%5010, %5053) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %5055 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5056 = "ttir.reshape"(%5054, %5055) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5057 = ttir.empty() : tensor<1x24x640x640xf32>
    %5058 = "ttir.typecast"(%5056, %5057) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5059 = ttir.empty() : tensor<1x24x640x640xf32>
    %5060 = "ttir.multiply"(%5058, %221, %5059) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5061 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5062 = "ttir.typecast"(%5060, %5061) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5063 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5064 = "ttir.add"(%5062, %285, %5063) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5065 = ttir.empty() : tensor<1x24x640x640xf32>
    %5066 = "ttir.typecast"(%5064, %5065) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5067 = ttir.empty() : tensor<1x24x640xf32>
    %5068 = "ttir.max"(%5066, %5067) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %5069 = ttir.empty() : tensor<1x24x640x1xf32>
    %5070 = "ttir.reshape"(%5068, %5069) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %5071 = ttir.empty() : tensor<1x24x640x640xf32>
    %5072 = "ttir.broadcast"(%5070, %5071) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5073 = ttir.empty() : tensor<1x24x640x640xf32>
    %5074 = "ttir.subtract"(%5066, %5072, %5073) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5075 = ttir.empty() : tensor<1x24x640x640xf32>
    %5076 = "ttir.exp"(%5074, %5075) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5077 = ttir.empty() : tensor<1x24x640xf32>
    %5078 = "ttir.sum"(%5076, %5077) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %5079 = ttir.empty() : tensor<1x24x640x1xf32>
    %5080 = "ttir.reshape"(%5078, %5079) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %5081 = ttir.empty() : tensor<1x24x640x640xf32>
    %5082 = "ttir.broadcast"(%5080, %5081) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5083 = ttir.empty() : tensor<1x24x640x640xf32>
    %5084 = "ttir.div"(%5076, %5082, %5083) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5085 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5086 = "ttir.typecast"(%5084, %5085) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5087 = ttir.empty() : tensor<24x640x640xbf16>
    %5088 = "ttir.reshape"(%5086, %5087) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %5089 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %5090 = "ttir.reshape"(%4965, %5089) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %5091 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %5092 = "ttir.broadcast"(%5090, %5091) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %5093 = ttir.empty() : tensor<24x640x128xbf16>
    %5094 = "ttir.reshape"(%5092, %5093) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5095 = "ttir.dot_general"(%5088, %5094) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5096 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5097 = "ttir.reshape"(%5095, %5096) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5098 = ttir.empty() : tensor<1x640x24x128xbf16>
    %5099 = "ttir.permute"(%5097, %5098) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %5100 = ttir.empty() : tensor<640x3072xbf16>
    %5101 = "ttir.reshape"(%5099, %5100) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5102 = ttir.empty() : tensor<1x3072x3072xbf16>
    %5103 = "ttir.reshape"(%arg165, %5102) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %5104 = ttir.empty() : tensor<3072x3072xbf16>
    %5105 = "ttir.reshape"(%5103, %5104) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5106 = ttir.empty() : tensor<3072x3072xbf16>
    %5107 = "ttir.permute"(%5105, %5106) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5108 = "ttir.dot_general"(%5101, %5107) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %5109 = ttir.empty() : tensor<1x640x3072xbf16>
    %5110 = "ttir.reshape"(%5108, %5109) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5111 = ttir.empty() : tensor<1x640x3072xbf16>
    %5112 = "ttir.add"(%4922, %5110, %5111) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5113 = ttir.empty() : tensor<1x1x3072xbf16>
    %5114 = "ttir.reshape"(%arg168, %5113) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %5115 = ttir.empty() : tensor<3072xbf16>
    %5116 = "ttir.reshape"(%5114, %5115) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %5117 = ttir.empty() : tensor<3072xf32>
    %5118 = "ttir.typecast"(%5116, %5117) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %5119 = ttir.empty() : tensor<1x1x3072xf32>
    %5120 = "ttir.reshape"(%5118, %5119) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %5121 = ttir.empty() : tensor<1x640x3072xf32>
    %5122 = "ttir.broadcast"(%5120, %5121) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5123 = ttir.empty() : tensor<1x640x3072xf32>
    %5124 = "ttir.typecast"(%5112, %5123) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5125 = ttir.empty() : tensor<1x640x3072xf32>
    %5126 = "ttir.pow"(%5124, %5, %5125) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5127 = ttir.empty() : tensor<1x640xf32>
    %5128 = "ttir.sum"(%5126, %5127) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5129 = ttir.empty() : tensor<1x640xf32>
    %5130 = "ttir.multiply"(%5128, %4, %5129) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5131 = ttir.empty() : tensor<1x640x1xf32>
    %5132 = "ttir.reshape"(%5130, %5131) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5133 = ttir.empty() : tensor<1x640x1xf32>
    %5134 = "ttir.add"(%5132, %46, %5133) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5135 = ttir.empty() : tensor<1x640x1xf32>
    %5136 = "ttir.rsqrt"(%5134, %5135) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5137 = ttir.empty() : tensor<1x640xf32>
    %5138 = "ttir.reshape"(%5136, %5137) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5139 = ttir.empty() : tensor<1x640x1xf32>
    %5140 = "ttir.reshape"(%5138, %5139) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5141 = ttir.empty() : tensor<1x640x3072xf32>
    %5142 = "ttir.broadcast"(%5140, %5141) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5143 = ttir.empty() : tensor<1x640x3072xf32>
    %5144 = "ttir.multiply"(%5124, %5142, %5143) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5145 = ttir.empty() : tensor<1x640x3072xbf16>
    %5146 = "ttir.typecast"(%5144, %5145) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5147 = ttir.empty() : tensor<1x640x3072xf32>
    %5148 = "ttir.typecast"(%5146, %5147) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5149 = ttir.empty() : tensor<1x640x3072xf32>
    %5150 = "ttir.multiply"(%5122, %5148, %5149) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5151 = ttir.empty() : tensor<1x640x3072xbf16>
    %5152 = "ttir.typecast"(%5150, %5151) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5153 = ttir.empty() : tensor<640x3072xbf16>
    %5154 = "ttir.reshape"(%5152, %5153) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5155 = ttir.empty() : tensor<1x8192x3072xbf16>
    %5156 = "ttir.reshape"(%arg169, %5155) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %5157 = ttir.empty() : tensor<8192x3072xbf16>
    %5158 = "ttir.reshape"(%5156, %5157) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %5159 = ttir.empty() : tensor<3072x8192xbf16>
    %5160 = "ttir.permute"(%5158, %5159) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %5161 = "ttir.dot_general"(%5154, %5160) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %5162 = ttir.empty() : tensor<1x640x8192xbf16>
    %5163 = "ttir.reshape"(%5161, %5162) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5164 = ttir.empty() : tensor<1x640x8192xf32>
    %5165 = "ttir.typecast"(%5163, %5164) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5166 = ttir.empty() : tensor<1x640x8192xbf16>
    %5167 = "ttir.sigmoid"(%5163, %5166) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5168 = ttir.empty() : tensor<1x640x8192xf32>
    %5169 = "ttir.typecast"(%5167, %5168) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5170 = ttir.empty() : tensor<1x640x8192xf32>
    %5171 = "ttir.multiply"(%5165, %5169, %5170) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5172 = ttir.empty() : tensor<1x640x8192xbf16>
    %5173 = "ttir.typecast"(%5171, %5172) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5174 = ttir.empty() : tensor<1x640x8192xf32>
    %5175 = "ttir.typecast"(%5173, %5174) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5176 = ttir.empty() : tensor<1x8192x3072xbf16>
    %5177 = "ttir.reshape"(%arg164, %5176) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %5178 = ttir.empty() : tensor<8192x3072xbf16>
    %5179 = "ttir.reshape"(%5177, %5178) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %5180 = ttir.empty() : tensor<3072x8192xbf16>
    %5181 = "ttir.permute"(%5179, %5180) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %5182 = "ttir.dot_general"(%5154, %5181) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %5183 = ttir.empty() : tensor<1x640x8192xbf16>
    %5184 = "ttir.reshape"(%5182, %5183) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5185 = ttir.empty() : tensor<1x640x8192xf32>
    %5186 = "ttir.typecast"(%5184, %5185) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5187 = ttir.empty() : tensor<1x640x8192xf32>
    %5188 = "ttir.multiply"(%5175, %5186, %5187) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5189 = ttir.empty() : tensor<1x640x8192xbf16>
    %5190 = "ttir.typecast"(%5188, %5189) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5191 = ttir.empty() : tensor<640x8192xbf16>
    %5192 = "ttir.reshape"(%5190, %5191) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %5193 = ttir.empty() : tensor<1x3072x8192xbf16>
    %5194 = "ttir.reshape"(%arg163, %5193) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %5195 = ttir.empty() : tensor<3072x8192xbf16>
    %5196 = "ttir.reshape"(%5194, %5195) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %5197 = ttir.empty() : tensor<8192x3072xbf16>
    %5198 = "ttir.permute"(%5196, %5197) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %5199 = "ttir.dot_general"(%5192, %5198) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %5200 = ttir.empty() : tensor<1x640x3072xbf16>
    %5201 = "ttir.reshape"(%5199, %5200) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5202 = ttir.empty() : tensor<1x640x3072xbf16>
    %5203 = "ttir.add"(%5112, %5201, %5202) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5204 = ttir.empty() : tensor<1x640x3072xf32>
    %5205 = "ttir.typecast"(%5203, %5204) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5206 = ttir.empty() : tensor<1x640x3072xf32>
    %5207 = "ttir.pow"(%5205, %5, %5206) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5208 = ttir.empty() : tensor<1x640xf32>
    %5209 = "ttir.sum"(%5207, %5208) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5210 = ttir.empty() : tensor<1x640xf32>
    %5211 = "ttir.multiply"(%5209, %4, %5210) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5212 = ttir.empty() : tensor<1x640x1xf32>
    %5213 = "ttir.reshape"(%5211, %5212) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5214 = ttir.empty() : tensor<1x640x1xf32>
    %5215 = "ttir.add"(%5213, %46, %5214) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5216 = ttir.empty() : tensor<1x640x1xf32>
    %5217 = "ttir.rsqrt"(%5215, %5216) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5218 = ttir.empty() : tensor<1x640xf32>
    %5219 = "ttir.reshape"(%5217, %5218) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5220 = ttir.empty() : tensor<1x640x1xf32>
    %5221 = "ttir.reshape"(%5219, %5220) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5222 = ttir.empty() : tensor<1x640x3072xf32>
    %5223 = "ttir.broadcast"(%5221, %5222) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5224 = ttir.empty() : tensor<1x640x3072xf32>
    %5225 = "ttir.multiply"(%5205, %5223, %5224) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5226 = ttir.empty() : tensor<1x640x3072xbf16>
    %5227 = "ttir.typecast"(%5225, %5226) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5228 = ttir.empty() : tensor<1x640x3072xf32>
    %5229 = "ttir.typecast"(%5227, %5228) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5230 = ttir.empty() : tensor<1x640x3072xf32>
    %5231 = "ttir.multiply"(%4975, %5229, %5230) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5232 = ttir.empty() : tensor<1x640x3072xbf16>
    %5233 = "ttir.typecast"(%5231, %5232) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5234 = ttir.empty() : tensor<640x3072xbf16>
    %5235 = "ttir.reshape"(%5233, %5234) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5236 = ttir.empty() : tensor<1x1024x3072xbf16>
    %5237 = "ttir.reshape"(%arg162, %5236) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %5238 = ttir.empty() : tensor<1024x3072xbf16>
    %5239 = "ttir.reshape"(%5237, %5238) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %5240 = ttir.empty() : tensor<3072x1024xbf16>
    %5241 = "ttir.permute"(%5239, %5240) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %5242 = "ttir.dot_general"(%5235, %5241) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %5243 = ttir.empty() : tensor<1x640x8x128xbf16>
    %5244 = "ttir.reshape"(%5242, %5243) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %5245 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5246 = "ttir.permute"(%5244, %5245) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5247 = ttir.empty() : tensor<1x1x3072xbf16>
    %5248 = "ttir.reshape"(%arg179, %5247) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %5249 = ttir.empty() : tensor<3072xbf16>
    %5250 = "ttir.reshape"(%5248, %5249) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %5251 = ttir.empty() : tensor<3072xf32>
    %5252 = "ttir.typecast"(%5250, %5251) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %5253 = ttir.empty() : tensor<1x1x3072xf32>
    %5254 = "ttir.reshape"(%5252, %5253) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %5255 = ttir.empty() : tensor<1x640x3072xf32>
    %5256 = "ttir.broadcast"(%5254, %5255) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5257 = ttir.empty() : tensor<1x3072x3072xbf16>
    %5258 = "ttir.reshape"(%arg176, %5257) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %5259 = ttir.empty() : tensor<3072x3072xbf16>
    %5260 = "ttir.reshape"(%5258, %5259) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5261 = ttir.empty() : tensor<3072x3072xbf16>
    %5262 = "ttir.permute"(%5260, %5261) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5263 = "ttir.dot_general"(%5235, %5262) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %5264 = ttir.empty() : tensor<1x640x24x128xbf16>
    %5265 = "ttir.reshape"(%5263, %5264) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %5266 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5267 = "ttir.permute"(%5265, %5266) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5268 = ttir.empty() : tensor<1x24x640x128xf32>
    %5269 = "ttir.typecast"(%5267, %5268) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5270 = ttir.empty() : tensor<1x24x640x128xf32>
    %5271 = "ttir.multiply"(%5269, %125, %5270) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5272 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5273 = "ttir.typecast"(%5271, %5272) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5274 = ttir.empty() : tensor<1x24x640x64xbf16>
    %5275 = "ttir.slice_static"(%5267, %5274) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %5276 = ttir.empty() : tensor<1x24x640x64xbf16>
    %5277 = "ttir.neg"(%5275, %5276) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %5278 = ttir.empty() : tensor<1x24x640x64xbf16>
    %5279 = "ttir.slice_static"(%5267, %5278) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %5280 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5281 = "ttir.concat"(%5277, %5279, %5280) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5282 = ttir.empty() : tensor<1x24x640x128xf32>
    %5283 = "ttir.typecast"(%5281, %5282) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5284 = ttir.empty() : tensor<1x24x640x128xf32>
    %5285 = "ttir.multiply"(%5283, %153, %5284) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5286 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5287 = "ttir.typecast"(%5285, %5286) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5288 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5289 = "ttir.add"(%5273, %5287, %5288) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5290 = ttir.empty() : tensor<24x640x128xbf16>
    %5291 = "ttir.reshape"(%5289, %5290) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5292 = ttir.empty() : tensor<1x1024x3072xbf16>
    %5293 = "ttir.reshape"(%arg175, %5292) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %5294 = ttir.empty() : tensor<1024x3072xbf16>
    %5295 = "ttir.reshape"(%5293, %5294) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %5296 = ttir.empty() : tensor<3072x1024xbf16>
    %5297 = "ttir.permute"(%5295, %5296) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %5298 = "ttir.dot_general"(%5235, %5297) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %5299 = ttir.empty() : tensor<1x640x8x128xbf16>
    %5300 = "ttir.reshape"(%5298, %5299) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %5301 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5302 = "ttir.permute"(%5300, %5301) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5303 = ttir.empty() : tensor<1x8x640x128xf32>
    %5304 = "ttir.typecast"(%5302, %5303) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5305 = ttir.empty() : tensor<1x8x640x128xf32>
    %5306 = "ttir.multiply"(%5304, %178, %5305) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5307 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5308 = "ttir.typecast"(%5306, %5307) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5309 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5310 = "ttir.slice_static"(%5302, %5309) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5311 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5312 = "ttir.neg"(%5310, %5311) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5313 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5314 = "ttir.slice_static"(%5302, %5313) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5315 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5316 = "ttir.concat"(%5312, %5314, %5315) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5317 = ttir.empty() : tensor<1x8x640x128xf32>
    %5318 = "ttir.typecast"(%5316, %5317) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5319 = ttir.empty() : tensor<1x8x640x128xf32>
    %5320 = "ttir.multiply"(%5318, %196, %5319) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5321 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5322 = "ttir.typecast"(%5320, %5321) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5323 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5324 = "ttir.add"(%5308, %5322, %5323) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5325 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %5326 = "ttir.reshape"(%5324, %5325) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %5327 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %5328 = "ttir.broadcast"(%5326, %5327) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %5329 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5330 = "ttir.reshape"(%5328, %5329) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5331 = ttir.empty() : tensor<1x24x128x640xbf16>
    %5332 = "ttir.permute"(%5330, %5331) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %5333 = ttir.empty() : tensor<24x128x640xbf16>
    %5334 = "ttir.reshape"(%5332, %5333) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %5335 = "ttir.dot_general"(%5291, %5334) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %5336 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5337 = "ttir.reshape"(%5335, %5336) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5338 = ttir.empty() : tensor<1x24x640x640xf32>
    %5339 = "ttir.typecast"(%5337, %5338) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5340 = ttir.empty() : tensor<1x24x640x640xf32>
    %5341 = "ttir.multiply"(%5339, %221, %5340) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5342 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5343 = "ttir.typecast"(%5341, %5342) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5344 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5345 = "ttir.add"(%5343, %285, %5344) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5346 = ttir.empty() : tensor<1x24x640x640xf32>
    %5347 = "ttir.typecast"(%5345, %5346) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5348 = ttir.empty() : tensor<1x24x640xf32>
    %5349 = "ttir.max"(%5347, %5348) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %5350 = ttir.empty() : tensor<1x24x640x1xf32>
    %5351 = "ttir.reshape"(%5349, %5350) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %5352 = ttir.empty() : tensor<1x24x640x640xf32>
    %5353 = "ttir.broadcast"(%5351, %5352) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5354 = ttir.empty() : tensor<1x24x640x640xf32>
    %5355 = "ttir.subtract"(%5347, %5353, %5354) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5356 = ttir.empty() : tensor<1x24x640x640xf32>
    %5357 = "ttir.exp"(%5355, %5356) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5358 = ttir.empty() : tensor<1x24x640xf32>
    %5359 = "ttir.sum"(%5357, %5358) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %5360 = ttir.empty() : tensor<1x24x640x1xf32>
    %5361 = "ttir.reshape"(%5359, %5360) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %5362 = ttir.empty() : tensor<1x24x640x640xf32>
    %5363 = "ttir.broadcast"(%5361, %5362) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5364 = ttir.empty() : tensor<1x24x640x640xf32>
    %5365 = "ttir.div"(%5357, %5363, %5364) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5366 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5367 = "ttir.typecast"(%5365, %5366) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5368 = ttir.empty() : tensor<24x640x640xbf16>
    %5369 = "ttir.reshape"(%5367, %5368) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %5370 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %5371 = "ttir.reshape"(%5246, %5370) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %5372 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %5373 = "ttir.broadcast"(%5371, %5372) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %5374 = ttir.empty() : tensor<24x640x128xbf16>
    %5375 = "ttir.reshape"(%5373, %5374) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5376 = "ttir.dot_general"(%5369, %5375) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5377 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5378 = "ttir.reshape"(%5376, %5377) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5379 = ttir.empty() : tensor<1x640x24x128xbf16>
    %5380 = "ttir.permute"(%5378, %5379) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %5381 = ttir.empty() : tensor<640x3072xbf16>
    %5382 = "ttir.reshape"(%5380, %5381) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5383 = ttir.empty() : tensor<1x3072x3072xbf16>
    %5384 = "ttir.reshape"(%arg174, %5383) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %5385 = ttir.empty() : tensor<3072x3072xbf16>
    %5386 = "ttir.reshape"(%5384, %5385) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5387 = ttir.empty() : tensor<3072x3072xbf16>
    %5388 = "ttir.permute"(%5386, %5387) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5389 = "ttir.dot_general"(%5382, %5388) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %5390 = ttir.empty() : tensor<1x640x3072xbf16>
    %5391 = "ttir.reshape"(%5389, %5390) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5392 = ttir.empty() : tensor<1x640x3072xbf16>
    %5393 = "ttir.add"(%5203, %5391, %5392) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5394 = ttir.empty() : tensor<1x1x3072xbf16>
    %5395 = "ttir.reshape"(%arg177, %5394) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %5396 = ttir.empty() : tensor<3072xbf16>
    %5397 = "ttir.reshape"(%5395, %5396) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %5398 = ttir.empty() : tensor<3072xf32>
    %5399 = "ttir.typecast"(%5397, %5398) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %5400 = ttir.empty() : tensor<1x1x3072xf32>
    %5401 = "ttir.reshape"(%5399, %5400) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %5402 = ttir.empty() : tensor<1x640x3072xf32>
    %5403 = "ttir.broadcast"(%5401, %5402) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5404 = ttir.empty() : tensor<1x640x3072xf32>
    %5405 = "ttir.typecast"(%5393, %5404) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5406 = ttir.empty() : tensor<1x640x3072xf32>
    %5407 = "ttir.pow"(%5405, %5, %5406) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5408 = ttir.empty() : tensor<1x640xf32>
    %5409 = "ttir.sum"(%5407, %5408) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5410 = ttir.empty() : tensor<1x640xf32>
    %5411 = "ttir.multiply"(%5409, %4, %5410) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5412 = ttir.empty() : tensor<1x640x1xf32>
    %5413 = "ttir.reshape"(%5411, %5412) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5414 = ttir.empty() : tensor<1x640x1xf32>
    %5415 = "ttir.add"(%5413, %46, %5414) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5416 = ttir.empty() : tensor<1x640x1xf32>
    %5417 = "ttir.rsqrt"(%5415, %5416) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5418 = ttir.empty() : tensor<1x640xf32>
    %5419 = "ttir.reshape"(%5417, %5418) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5420 = ttir.empty() : tensor<1x640x1xf32>
    %5421 = "ttir.reshape"(%5419, %5420) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5422 = ttir.empty() : tensor<1x640x3072xf32>
    %5423 = "ttir.broadcast"(%5421, %5422) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5424 = ttir.empty() : tensor<1x640x3072xf32>
    %5425 = "ttir.multiply"(%5405, %5423, %5424) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5426 = ttir.empty() : tensor<1x640x3072xbf16>
    %5427 = "ttir.typecast"(%5425, %5426) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5428 = ttir.empty() : tensor<1x640x3072xf32>
    %5429 = "ttir.typecast"(%5427, %5428) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5430 = ttir.empty() : tensor<1x640x3072xf32>
    %5431 = "ttir.multiply"(%5403, %5429, %5430) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5432 = ttir.empty() : tensor<1x640x3072xbf16>
    %5433 = "ttir.typecast"(%5431, %5432) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5434 = ttir.empty() : tensor<640x3072xbf16>
    %5435 = "ttir.reshape"(%5433, %5434) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5436 = ttir.empty() : tensor<1x8192x3072xbf16>
    %5437 = "ttir.reshape"(%arg178, %5436) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %5438 = ttir.empty() : tensor<8192x3072xbf16>
    %5439 = "ttir.reshape"(%5437, %5438) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %5440 = ttir.empty() : tensor<3072x8192xbf16>
    %5441 = "ttir.permute"(%5439, %5440) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %5442 = "ttir.dot_general"(%5435, %5441) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %5443 = ttir.empty() : tensor<1x640x8192xbf16>
    %5444 = "ttir.reshape"(%5442, %5443) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5445 = ttir.empty() : tensor<1x640x8192xf32>
    %5446 = "ttir.typecast"(%5444, %5445) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5447 = ttir.empty() : tensor<1x640x8192xbf16>
    %5448 = "ttir.sigmoid"(%5444, %5447) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5449 = ttir.empty() : tensor<1x640x8192xf32>
    %5450 = "ttir.typecast"(%5448, %5449) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5451 = ttir.empty() : tensor<1x640x8192xf32>
    %5452 = "ttir.multiply"(%5446, %5450, %5451) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5453 = ttir.empty() : tensor<1x640x8192xbf16>
    %5454 = "ttir.typecast"(%5452, %5453) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5455 = ttir.empty() : tensor<1x640x8192xf32>
    %5456 = "ttir.typecast"(%5454, %5455) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5457 = ttir.empty() : tensor<1x8192x3072xbf16>
    %5458 = "ttir.reshape"(%arg173, %5457) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %5459 = ttir.empty() : tensor<8192x3072xbf16>
    %5460 = "ttir.reshape"(%5458, %5459) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %5461 = ttir.empty() : tensor<3072x8192xbf16>
    %5462 = "ttir.permute"(%5460, %5461) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %5463 = "ttir.dot_general"(%5435, %5462) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %5464 = ttir.empty() : tensor<1x640x8192xbf16>
    %5465 = "ttir.reshape"(%5463, %5464) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5466 = ttir.empty() : tensor<1x640x8192xf32>
    %5467 = "ttir.typecast"(%5465, %5466) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5468 = ttir.empty() : tensor<1x640x8192xf32>
    %5469 = "ttir.multiply"(%5456, %5467, %5468) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5470 = ttir.empty() : tensor<1x640x8192xbf16>
    %5471 = "ttir.typecast"(%5469, %5470) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5472 = ttir.empty() : tensor<640x8192xbf16>
    %5473 = "ttir.reshape"(%5471, %5472) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %5474 = ttir.empty() : tensor<1x3072x8192xbf16>
    %5475 = "ttir.reshape"(%arg172, %5474) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %5476 = ttir.empty() : tensor<3072x8192xbf16>
    %5477 = "ttir.reshape"(%5475, %5476) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %5478 = ttir.empty() : tensor<8192x3072xbf16>
    %5479 = "ttir.permute"(%5477, %5478) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %5480 = "ttir.dot_general"(%5473, %5479) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %5481 = ttir.empty() : tensor<1x640x3072xbf16>
    %5482 = "ttir.reshape"(%5480, %5481) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5483 = ttir.empty() : tensor<1x640x3072xbf16>
    %5484 = "ttir.add"(%5393, %5482, %5483) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5485 = ttir.empty() : tensor<1x640x3072xf32>
    %5486 = "ttir.typecast"(%5484, %5485) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5487 = ttir.empty() : tensor<1x640x3072xf32>
    %5488 = "ttir.pow"(%5486, %5, %5487) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5489 = ttir.empty() : tensor<1x640xf32>
    %5490 = "ttir.sum"(%5488, %5489) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5491 = ttir.empty() : tensor<1x640xf32>
    %5492 = "ttir.multiply"(%5490, %4, %5491) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5493 = ttir.empty() : tensor<1x640x1xf32>
    %5494 = "ttir.reshape"(%5492, %5493) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5495 = ttir.empty() : tensor<1x640x1xf32>
    %5496 = "ttir.add"(%5494, %46, %5495) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5497 = ttir.empty() : tensor<1x640x1xf32>
    %5498 = "ttir.rsqrt"(%5496, %5497) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5499 = ttir.empty() : tensor<1x640xf32>
    %5500 = "ttir.reshape"(%5498, %5499) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5501 = ttir.empty() : tensor<1x640x1xf32>
    %5502 = "ttir.reshape"(%5500, %5501) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5503 = ttir.empty() : tensor<1x640x3072xf32>
    %5504 = "ttir.broadcast"(%5502, %5503) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5505 = ttir.empty() : tensor<1x640x3072xf32>
    %5506 = "ttir.multiply"(%5486, %5504, %5505) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5507 = ttir.empty() : tensor<1x640x3072xbf16>
    %5508 = "ttir.typecast"(%5506, %5507) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5509 = ttir.empty() : tensor<1x640x3072xf32>
    %5510 = "ttir.typecast"(%5508, %5509) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5511 = ttir.empty() : tensor<1x640x3072xf32>
    %5512 = "ttir.multiply"(%5256, %5510, %5511) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5513 = ttir.empty() : tensor<1x640x3072xbf16>
    %5514 = "ttir.typecast"(%5512, %5513) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5515 = ttir.empty() : tensor<640x3072xbf16>
    %5516 = "ttir.reshape"(%5514, %5515) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5517 = ttir.empty() : tensor<1x1024x3072xbf16>
    %5518 = "ttir.reshape"(%arg171, %5517) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %5519 = ttir.empty() : tensor<1024x3072xbf16>
    %5520 = "ttir.reshape"(%5518, %5519) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %5521 = ttir.empty() : tensor<3072x1024xbf16>
    %5522 = "ttir.permute"(%5520, %5521) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %5523 = "ttir.dot_general"(%5516, %5522) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %5524 = ttir.empty() : tensor<1x640x8x128xbf16>
    %5525 = "ttir.reshape"(%5523, %5524) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %5526 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5527 = "ttir.permute"(%5525, %5526) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5528 = ttir.empty() : tensor<1x1x3072xbf16>
    %5529 = "ttir.reshape"(%arg188, %5528) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %5530 = ttir.empty() : tensor<3072xbf16>
    %5531 = "ttir.reshape"(%5529, %5530) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %5532 = ttir.empty() : tensor<3072xf32>
    %5533 = "ttir.typecast"(%5531, %5532) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %5534 = ttir.empty() : tensor<1x1x3072xf32>
    %5535 = "ttir.reshape"(%5533, %5534) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %5536 = ttir.empty() : tensor<1x640x3072xf32>
    %5537 = "ttir.broadcast"(%5535, %5536) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5538 = ttir.empty() : tensor<1x3072x3072xbf16>
    %5539 = "ttir.reshape"(%arg185, %5538) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %5540 = ttir.empty() : tensor<3072x3072xbf16>
    %5541 = "ttir.reshape"(%5539, %5540) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5542 = ttir.empty() : tensor<3072x3072xbf16>
    %5543 = "ttir.permute"(%5541, %5542) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5544 = "ttir.dot_general"(%5516, %5543) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %5545 = ttir.empty() : tensor<1x640x24x128xbf16>
    %5546 = "ttir.reshape"(%5544, %5545) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %5547 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5548 = "ttir.permute"(%5546, %5547) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5549 = ttir.empty() : tensor<1x24x640x128xf32>
    %5550 = "ttir.typecast"(%5548, %5549) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5551 = ttir.empty() : tensor<1x24x640x128xf32>
    %5552 = "ttir.multiply"(%5550, %125, %5551) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5553 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5554 = "ttir.typecast"(%5552, %5553) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5555 = ttir.empty() : tensor<1x24x640x64xbf16>
    %5556 = "ttir.slice_static"(%5548, %5555) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %5557 = ttir.empty() : tensor<1x24x640x64xbf16>
    %5558 = "ttir.neg"(%5556, %5557) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %5559 = ttir.empty() : tensor<1x24x640x64xbf16>
    %5560 = "ttir.slice_static"(%5548, %5559) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %5561 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5562 = "ttir.concat"(%5558, %5560, %5561) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5563 = ttir.empty() : tensor<1x24x640x128xf32>
    %5564 = "ttir.typecast"(%5562, %5563) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5565 = ttir.empty() : tensor<1x24x640x128xf32>
    %5566 = "ttir.multiply"(%5564, %153, %5565) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5567 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5568 = "ttir.typecast"(%5566, %5567) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5569 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5570 = "ttir.add"(%5554, %5568, %5569) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5571 = ttir.empty() : tensor<24x640x128xbf16>
    %5572 = "ttir.reshape"(%5570, %5571) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5573 = ttir.empty() : tensor<1x1024x3072xbf16>
    %5574 = "ttir.reshape"(%arg184, %5573) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %5575 = ttir.empty() : tensor<1024x3072xbf16>
    %5576 = "ttir.reshape"(%5574, %5575) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %5577 = ttir.empty() : tensor<3072x1024xbf16>
    %5578 = "ttir.permute"(%5576, %5577) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %5579 = "ttir.dot_general"(%5516, %5578) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %5580 = ttir.empty() : tensor<1x640x8x128xbf16>
    %5581 = "ttir.reshape"(%5579, %5580) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %5582 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5583 = "ttir.permute"(%5581, %5582) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5584 = ttir.empty() : tensor<1x8x640x128xf32>
    %5585 = "ttir.typecast"(%5583, %5584) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5586 = ttir.empty() : tensor<1x8x640x128xf32>
    %5587 = "ttir.multiply"(%5585, %178, %5586) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5588 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5589 = "ttir.typecast"(%5587, %5588) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5590 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5591 = "ttir.slice_static"(%5583, %5590) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5592 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5593 = "ttir.neg"(%5591, %5592) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5594 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5595 = "ttir.slice_static"(%5583, %5594) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5596 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5597 = "ttir.concat"(%5593, %5595, %5596) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5598 = ttir.empty() : tensor<1x8x640x128xf32>
    %5599 = "ttir.typecast"(%5597, %5598) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5600 = ttir.empty() : tensor<1x8x640x128xf32>
    %5601 = "ttir.multiply"(%5599, %196, %5600) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5602 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5603 = "ttir.typecast"(%5601, %5602) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5604 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5605 = "ttir.add"(%5589, %5603, %5604) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5606 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %5607 = "ttir.reshape"(%5605, %5606) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %5608 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %5609 = "ttir.broadcast"(%5607, %5608) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %5610 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5611 = "ttir.reshape"(%5609, %5610) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5612 = ttir.empty() : tensor<1x24x128x640xbf16>
    %5613 = "ttir.permute"(%5611, %5612) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %5614 = ttir.empty() : tensor<24x128x640xbf16>
    %5615 = "ttir.reshape"(%5613, %5614) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %5616 = "ttir.dot_general"(%5572, %5615) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %5617 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5618 = "ttir.reshape"(%5616, %5617) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5619 = ttir.empty() : tensor<1x24x640x640xf32>
    %5620 = "ttir.typecast"(%5618, %5619) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5621 = ttir.empty() : tensor<1x24x640x640xf32>
    %5622 = "ttir.multiply"(%5620, %221, %5621) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5623 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5624 = "ttir.typecast"(%5622, %5623) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5625 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5626 = "ttir.add"(%5624, %285, %5625) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5627 = ttir.empty() : tensor<1x24x640x640xf32>
    %5628 = "ttir.typecast"(%5626, %5627) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5629 = ttir.empty() : tensor<1x24x640xf32>
    %5630 = "ttir.max"(%5628, %5629) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %5631 = ttir.empty() : tensor<1x24x640x1xf32>
    %5632 = "ttir.reshape"(%5630, %5631) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %5633 = ttir.empty() : tensor<1x24x640x640xf32>
    %5634 = "ttir.broadcast"(%5632, %5633) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5635 = ttir.empty() : tensor<1x24x640x640xf32>
    %5636 = "ttir.subtract"(%5628, %5634, %5635) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5637 = ttir.empty() : tensor<1x24x640x640xf32>
    %5638 = "ttir.exp"(%5636, %5637) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5639 = ttir.empty() : tensor<1x24x640xf32>
    %5640 = "ttir.sum"(%5638, %5639) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %5641 = ttir.empty() : tensor<1x24x640x1xf32>
    %5642 = "ttir.reshape"(%5640, %5641) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %5643 = ttir.empty() : tensor<1x24x640x640xf32>
    %5644 = "ttir.broadcast"(%5642, %5643) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5645 = ttir.empty() : tensor<1x24x640x640xf32>
    %5646 = "ttir.div"(%5638, %5644, %5645) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5647 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5648 = "ttir.typecast"(%5646, %5647) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5649 = ttir.empty() : tensor<24x640x640xbf16>
    %5650 = "ttir.reshape"(%5648, %5649) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %5651 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %5652 = "ttir.reshape"(%5527, %5651) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %5653 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %5654 = "ttir.broadcast"(%5652, %5653) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %5655 = ttir.empty() : tensor<24x640x128xbf16>
    %5656 = "ttir.reshape"(%5654, %5655) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5657 = "ttir.dot_general"(%5650, %5656) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5658 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5659 = "ttir.reshape"(%5657, %5658) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5660 = ttir.empty() : tensor<1x640x24x128xbf16>
    %5661 = "ttir.permute"(%5659, %5660) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %5662 = ttir.empty() : tensor<640x3072xbf16>
    %5663 = "ttir.reshape"(%5661, %5662) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5664 = ttir.empty() : tensor<1x3072x3072xbf16>
    %5665 = "ttir.reshape"(%arg183, %5664) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %5666 = ttir.empty() : tensor<3072x3072xbf16>
    %5667 = "ttir.reshape"(%5665, %5666) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5668 = ttir.empty() : tensor<3072x3072xbf16>
    %5669 = "ttir.permute"(%5667, %5668) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5670 = "ttir.dot_general"(%5663, %5669) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %5671 = ttir.empty() : tensor<1x640x3072xbf16>
    %5672 = "ttir.reshape"(%5670, %5671) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5673 = ttir.empty() : tensor<1x640x3072xbf16>
    %5674 = "ttir.add"(%5484, %5672, %5673) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5675 = ttir.empty() : tensor<1x1x3072xbf16>
    %5676 = "ttir.reshape"(%arg186, %5675) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %5677 = ttir.empty() : tensor<3072xbf16>
    %5678 = "ttir.reshape"(%5676, %5677) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %5679 = ttir.empty() : tensor<3072xf32>
    %5680 = "ttir.typecast"(%5678, %5679) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %5681 = ttir.empty() : tensor<1x1x3072xf32>
    %5682 = "ttir.reshape"(%5680, %5681) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %5683 = ttir.empty() : tensor<1x640x3072xf32>
    %5684 = "ttir.broadcast"(%5682, %5683) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5685 = ttir.empty() : tensor<1x640x3072xf32>
    %5686 = "ttir.typecast"(%5674, %5685) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5687 = ttir.empty() : tensor<1x640x3072xf32>
    %5688 = "ttir.pow"(%5686, %5, %5687) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5689 = ttir.empty() : tensor<1x640xf32>
    %5690 = "ttir.sum"(%5688, %5689) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5691 = ttir.empty() : tensor<1x640xf32>
    %5692 = "ttir.multiply"(%5690, %4, %5691) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5693 = ttir.empty() : tensor<1x640x1xf32>
    %5694 = "ttir.reshape"(%5692, %5693) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5695 = ttir.empty() : tensor<1x640x1xf32>
    %5696 = "ttir.add"(%5694, %46, %5695) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5697 = ttir.empty() : tensor<1x640x1xf32>
    %5698 = "ttir.rsqrt"(%5696, %5697) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5699 = ttir.empty() : tensor<1x640xf32>
    %5700 = "ttir.reshape"(%5698, %5699) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5701 = ttir.empty() : tensor<1x640x1xf32>
    %5702 = "ttir.reshape"(%5700, %5701) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5703 = ttir.empty() : tensor<1x640x3072xf32>
    %5704 = "ttir.broadcast"(%5702, %5703) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5705 = ttir.empty() : tensor<1x640x3072xf32>
    %5706 = "ttir.multiply"(%5686, %5704, %5705) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5707 = ttir.empty() : tensor<1x640x3072xbf16>
    %5708 = "ttir.typecast"(%5706, %5707) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5709 = ttir.empty() : tensor<1x640x3072xf32>
    %5710 = "ttir.typecast"(%5708, %5709) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5711 = ttir.empty() : tensor<1x640x3072xf32>
    %5712 = "ttir.multiply"(%5684, %5710, %5711) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5713 = ttir.empty() : tensor<1x640x3072xbf16>
    %5714 = "ttir.typecast"(%5712, %5713) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5715 = ttir.empty() : tensor<640x3072xbf16>
    %5716 = "ttir.reshape"(%5714, %5715) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5717 = ttir.empty() : tensor<1x8192x3072xbf16>
    %5718 = "ttir.reshape"(%arg187, %5717) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %5719 = ttir.empty() : tensor<8192x3072xbf16>
    %5720 = "ttir.reshape"(%5718, %5719) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %5721 = ttir.empty() : tensor<3072x8192xbf16>
    %5722 = "ttir.permute"(%5720, %5721) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %5723 = "ttir.dot_general"(%5716, %5722) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %5724 = ttir.empty() : tensor<1x640x8192xbf16>
    %5725 = "ttir.reshape"(%5723, %5724) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5726 = ttir.empty() : tensor<1x640x8192xf32>
    %5727 = "ttir.typecast"(%5725, %5726) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5728 = ttir.empty() : tensor<1x640x8192xbf16>
    %5729 = "ttir.sigmoid"(%5725, %5728) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5730 = ttir.empty() : tensor<1x640x8192xf32>
    %5731 = "ttir.typecast"(%5729, %5730) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5732 = ttir.empty() : tensor<1x640x8192xf32>
    %5733 = "ttir.multiply"(%5727, %5731, %5732) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5734 = ttir.empty() : tensor<1x640x8192xbf16>
    %5735 = "ttir.typecast"(%5733, %5734) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5736 = ttir.empty() : tensor<1x640x8192xf32>
    %5737 = "ttir.typecast"(%5735, %5736) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5738 = ttir.empty() : tensor<1x8192x3072xbf16>
    %5739 = "ttir.reshape"(%arg182, %5738) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %5740 = ttir.empty() : tensor<8192x3072xbf16>
    %5741 = "ttir.reshape"(%5739, %5740) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %5742 = ttir.empty() : tensor<3072x8192xbf16>
    %5743 = "ttir.permute"(%5741, %5742) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %5744 = "ttir.dot_general"(%5716, %5743) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %5745 = ttir.empty() : tensor<1x640x8192xbf16>
    %5746 = "ttir.reshape"(%5744, %5745) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5747 = ttir.empty() : tensor<1x640x8192xf32>
    %5748 = "ttir.typecast"(%5746, %5747) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5749 = ttir.empty() : tensor<1x640x8192xf32>
    %5750 = "ttir.multiply"(%5737, %5748, %5749) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %5751 = ttir.empty() : tensor<1x640x8192xbf16>
    %5752 = "ttir.typecast"(%5750, %5751) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %5753 = ttir.empty() : tensor<640x8192xbf16>
    %5754 = "ttir.reshape"(%5752, %5753) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %5755 = ttir.empty() : tensor<1x3072x8192xbf16>
    %5756 = "ttir.reshape"(%arg181, %5755) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %5757 = ttir.empty() : tensor<3072x8192xbf16>
    %5758 = "ttir.reshape"(%5756, %5757) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %5759 = ttir.empty() : tensor<8192x3072xbf16>
    %5760 = "ttir.permute"(%5758, %5759) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %5761 = "ttir.dot_general"(%5754, %5760) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %5762 = ttir.empty() : tensor<1x640x3072xbf16>
    %5763 = "ttir.reshape"(%5761, %5762) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5764 = ttir.empty() : tensor<1x640x3072xbf16>
    %5765 = "ttir.add"(%5674, %5763, %5764) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5766 = ttir.empty() : tensor<1x640x3072xf32>
    %5767 = "ttir.typecast"(%5765, %5766) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5768 = ttir.empty() : tensor<1x640x3072xf32>
    %5769 = "ttir.pow"(%5767, %5, %5768) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5770 = ttir.empty() : tensor<1x640xf32>
    %5771 = "ttir.sum"(%5769, %5770) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5772 = ttir.empty() : tensor<1x640xf32>
    %5773 = "ttir.multiply"(%5771, %4, %5772) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5774 = ttir.empty() : tensor<1x640x1xf32>
    %5775 = "ttir.reshape"(%5773, %5774) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5776 = ttir.empty() : tensor<1x640x1xf32>
    %5777 = "ttir.add"(%5775, %46, %5776) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5778 = ttir.empty() : tensor<1x640x1xf32>
    %5779 = "ttir.rsqrt"(%5777, %5778) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5780 = ttir.empty() : tensor<1x640xf32>
    %5781 = "ttir.reshape"(%5779, %5780) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5782 = ttir.empty() : tensor<1x640x1xf32>
    %5783 = "ttir.reshape"(%5781, %5782) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5784 = ttir.empty() : tensor<1x640x3072xf32>
    %5785 = "ttir.broadcast"(%5783, %5784) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5786 = ttir.empty() : tensor<1x640x3072xf32>
    %5787 = "ttir.multiply"(%5767, %5785, %5786) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5788 = ttir.empty() : tensor<1x640x3072xbf16>
    %5789 = "ttir.typecast"(%5787, %5788) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5790 = ttir.empty() : tensor<1x640x3072xf32>
    %5791 = "ttir.typecast"(%5789, %5790) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5792 = ttir.empty() : tensor<1x640x3072xf32>
    %5793 = "ttir.multiply"(%5537, %5791, %5792) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5794 = ttir.empty() : tensor<1x640x3072xbf16>
    %5795 = "ttir.typecast"(%5793, %5794) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5796 = ttir.empty() : tensor<640x3072xbf16>
    %5797 = "ttir.reshape"(%5795, %5796) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5798 = ttir.empty() : tensor<1x1024x3072xbf16>
    %5799 = "ttir.reshape"(%arg180, %5798) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %5800 = ttir.empty() : tensor<1024x3072xbf16>
    %5801 = "ttir.reshape"(%5799, %5800) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %5802 = ttir.empty() : tensor<3072x1024xbf16>
    %5803 = "ttir.permute"(%5801, %5802) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %5804 = "ttir.dot_general"(%5797, %5803) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %5805 = ttir.empty() : tensor<1x640x8x128xbf16>
    %5806 = "ttir.reshape"(%5804, %5805) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %5807 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5808 = "ttir.permute"(%5806, %5807) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5809 = ttir.empty() : tensor<1x1x3072xbf16>
    %5810 = "ttir.reshape"(%arg197, %5809) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %5811 = ttir.empty() : tensor<3072xbf16>
    %5812 = "ttir.reshape"(%5810, %5811) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %5813 = ttir.empty() : tensor<3072xf32>
    %5814 = "ttir.typecast"(%5812, %5813) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %5815 = ttir.empty() : tensor<1x1x3072xf32>
    %5816 = "ttir.reshape"(%5814, %5815) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %5817 = ttir.empty() : tensor<1x640x3072xf32>
    %5818 = "ttir.broadcast"(%5816, %5817) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5819 = ttir.empty() : tensor<1x3072x3072xbf16>
    %5820 = "ttir.reshape"(%arg194, %5819) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %5821 = ttir.empty() : tensor<3072x3072xbf16>
    %5822 = "ttir.reshape"(%5820, %5821) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5823 = ttir.empty() : tensor<3072x3072xbf16>
    %5824 = "ttir.permute"(%5822, %5823) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5825 = "ttir.dot_general"(%5797, %5824) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %5826 = ttir.empty() : tensor<1x640x24x128xbf16>
    %5827 = "ttir.reshape"(%5825, %5826) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %5828 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5829 = "ttir.permute"(%5827, %5828) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5830 = ttir.empty() : tensor<1x24x640x128xf32>
    %5831 = "ttir.typecast"(%5829, %5830) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5832 = ttir.empty() : tensor<1x24x640x128xf32>
    %5833 = "ttir.multiply"(%5831, %125, %5832) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5834 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5835 = "ttir.typecast"(%5833, %5834) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5836 = ttir.empty() : tensor<1x24x640x64xbf16>
    %5837 = "ttir.slice_static"(%5829, %5836) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %5838 = ttir.empty() : tensor<1x24x640x64xbf16>
    %5839 = "ttir.neg"(%5837, %5838) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %5840 = ttir.empty() : tensor<1x24x640x64xbf16>
    %5841 = "ttir.slice_static"(%5829, %5840) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %5842 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5843 = "ttir.concat"(%5839, %5841, %5842) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5844 = ttir.empty() : tensor<1x24x640x128xf32>
    %5845 = "ttir.typecast"(%5843, %5844) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5846 = ttir.empty() : tensor<1x24x640x128xf32>
    %5847 = "ttir.multiply"(%5845, %153, %5846) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %5848 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5849 = "ttir.typecast"(%5847, %5848) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5850 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5851 = "ttir.add"(%5835, %5849, %5850) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5852 = ttir.empty() : tensor<24x640x128xbf16>
    %5853 = "ttir.reshape"(%5851, %5852) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5854 = ttir.empty() : tensor<1x1024x3072xbf16>
    %5855 = "ttir.reshape"(%arg193, %5854) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %5856 = ttir.empty() : tensor<1024x3072xbf16>
    %5857 = "ttir.reshape"(%5855, %5856) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %5858 = ttir.empty() : tensor<3072x1024xbf16>
    %5859 = "ttir.permute"(%5857, %5858) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %5860 = "ttir.dot_general"(%5797, %5859) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %5861 = ttir.empty() : tensor<1x640x8x128xbf16>
    %5862 = "ttir.reshape"(%5860, %5861) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %5863 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5864 = "ttir.permute"(%5862, %5863) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5865 = ttir.empty() : tensor<1x8x640x128xf32>
    %5866 = "ttir.typecast"(%5864, %5865) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5867 = ttir.empty() : tensor<1x8x640x128xf32>
    %5868 = "ttir.multiply"(%5866, %178, %5867) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5869 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5870 = "ttir.typecast"(%5868, %5869) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5871 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5872 = "ttir.slice_static"(%5864, %5871) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5873 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5874 = "ttir.neg"(%5872, %5873) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5875 = ttir.empty() : tensor<1x8x640x64xbf16>
    %5876 = "ttir.slice_static"(%5864, %5875) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %5877 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5878 = "ttir.concat"(%5874, %5876, %5877) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5879 = ttir.empty() : tensor<1x8x640x128xf32>
    %5880 = "ttir.typecast"(%5878, %5879) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5881 = ttir.empty() : tensor<1x8x640x128xf32>
    %5882 = "ttir.multiply"(%5880, %196, %5881) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %5883 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5884 = "ttir.typecast"(%5882, %5883) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5885 = ttir.empty() : tensor<1x8x640x128xbf16>
    %5886 = "ttir.add"(%5870, %5884, %5885) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %5887 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %5888 = "ttir.reshape"(%5886, %5887) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %5889 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %5890 = "ttir.broadcast"(%5888, %5889) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %5891 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5892 = "ttir.reshape"(%5890, %5891) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5893 = ttir.empty() : tensor<1x24x128x640xbf16>
    %5894 = "ttir.permute"(%5892, %5893) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %5895 = ttir.empty() : tensor<24x128x640xbf16>
    %5896 = "ttir.reshape"(%5894, %5895) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %5897 = "ttir.dot_general"(%5853, %5896) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %5898 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5899 = "ttir.reshape"(%5897, %5898) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5900 = ttir.empty() : tensor<1x24x640x640xf32>
    %5901 = "ttir.typecast"(%5899, %5900) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5902 = ttir.empty() : tensor<1x24x640x640xf32>
    %5903 = "ttir.multiply"(%5901, %221, %5902) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5904 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5905 = "ttir.typecast"(%5903, %5904) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5906 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5907 = "ttir.add"(%5905, %285, %5906) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5908 = ttir.empty() : tensor<1x24x640x640xf32>
    %5909 = "ttir.typecast"(%5907, %5908) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5910 = ttir.empty() : tensor<1x24x640xf32>
    %5911 = "ttir.max"(%5909, %5910) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %5912 = ttir.empty() : tensor<1x24x640x1xf32>
    %5913 = "ttir.reshape"(%5911, %5912) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %5914 = ttir.empty() : tensor<1x24x640x640xf32>
    %5915 = "ttir.broadcast"(%5913, %5914) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5916 = ttir.empty() : tensor<1x24x640x640xf32>
    %5917 = "ttir.subtract"(%5909, %5915, %5916) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5918 = ttir.empty() : tensor<1x24x640x640xf32>
    %5919 = "ttir.exp"(%5917, %5918) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5920 = ttir.empty() : tensor<1x24x640xf32>
    %5921 = "ttir.sum"(%5919, %5920) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %5922 = ttir.empty() : tensor<1x24x640x1xf32>
    %5923 = "ttir.reshape"(%5921, %5922) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %5924 = ttir.empty() : tensor<1x24x640x640xf32>
    %5925 = "ttir.broadcast"(%5923, %5924) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5926 = ttir.empty() : tensor<1x24x640x640xf32>
    %5927 = "ttir.div"(%5919, %5925, %5926) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %5928 = ttir.empty() : tensor<1x24x640x640xbf16>
    %5929 = "ttir.typecast"(%5927, %5928) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %5930 = ttir.empty() : tensor<24x640x640xbf16>
    %5931 = "ttir.reshape"(%5929, %5930) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %5932 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %5933 = "ttir.reshape"(%5808, %5932) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %5934 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %5935 = "ttir.broadcast"(%5933, %5934) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %5936 = ttir.empty() : tensor<24x640x128xbf16>
    %5937 = "ttir.reshape"(%5935, %5936) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5938 = "ttir.dot_general"(%5931, %5937) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %5939 = ttir.empty() : tensor<1x24x640x128xbf16>
    %5940 = "ttir.reshape"(%5938, %5939) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %5941 = ttir.empty() : tensor<1x640x24x128xbf16>
    %5942 = "ttir.permute"(%5940, %5941) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %5943 = ttir.empty() : tensor<640x3072xbf16>
    %5944 = "ttir.reshape"(%5942, %5943) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5945 = ttir.empty() : tensor<1x3072x3072xbf16>
    %5946 = "ttir.reshape"(%arg192, %5945) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %5947 = ttir.empty() : tensor<3072x3072xbf16>
    %5948 = "ttir.reshape"(%5946, %5947) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5949 = ttir.empty() : tensor<3072x3072xbf16>
    %5950 = "ttir.permute"(%5948, %5949) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %5951 = "ttir.dot_general"(%5944, %5950) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %5952 = ttir.empty() : tensor<1x640x3072xbf16>
    %5953 = "ttir.reshape"(%5951, %5952) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5954 = ttir.empty() : tensor<1x640x3072xbf16>
    %5955 = "ttir.add"(%5765, %5953, %5954) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5956 = ttir.empty() : tensor<1x1x3072xbf16>
    %5957 = "ttir.reshape"(%arg195, %5956) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %5958 = ttir.empty() : tensor<3072xbf16>
    %5959 = "ttir.reshape"(%5957, %5958) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %5960 = ttir.empty() : tensor<3072xf32>
    %5961 = "ttir.typecast"(%5959, %5960) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %5962 = ttir.empty() : tensor<1x1x3072xf32>
    %5963 = "ttir.reshape"(%5961, %5962) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %5964 = ttir.empty() : tensor<1x640x3072xf32>
    %5965 = "ttir.broadcast"(%5963, %5964) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5966 = ttir.empty() : tensor<1x640x3072xf32>
    %5967 = "ttir.typecast"(%5955, %5966) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5968 = ttir.empty() : tensor<1x640x3072xf32>
    %5969 = "ttir.pow"(%5967, %5, %5968) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5970 = ttir.empty() : tensor<1x640xf32>
    %5971 = "ttir.sum"(%5969, %5970) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5972 = ttir.empty() : tensor<1x640xf32>
    %5973 = "ttir.multiply"(%5971, %4, %5972) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5974 = ttir.empty() : tensor<1x640x1xf32>
    %5975 = "ttir.reshape"(%5973, %5974) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5976 = ttir.empty() : tensor<1x640x1xf32>
    %5977 = "ttir.add"(%5975, %46, %5976) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5978 = ttir.empty() : tensor<1x640x1xf32>
    %5979 = "ttir.rsqrt"(%5977, %5978) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5980 = ttir.empty() : tensor<1x640xf32>
    %5981 = "ttir.reshape"(%5979, %5980) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5982 = ttir.empty() : tensor<1x640x1xf32>
    %5983 = "ttir.reshape"(%5981, %5982) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %5984 = ttir.empty() : tensor<1x640x3072xf32>
    %5985 = "ttir.broadcast"(%5983, %5984) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5986 = ttir.empty() : tensor<1x640x3072xf32>
    %5987 = "ttir.multiply"(%5967, %5985, %5986) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5988 = ttir.empty() : tensor<1x640x3072xbf16>
    %5989 = "ttir.typecast"(%5987, %5988) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5990 = ttir.empty() : tensor<1x640x3072xf32>
    %5991 = "ttir.typecast"(%5989, %5990) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5992 = ttir.empty() : tensor<1x640x3072xf32>
    %5993 = "ttir.multiply"(%5965, %5991, %5992) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %5994 = ttir.empty() : tensor<1x640x3072xbf16>
    %5995 = "ttir.typecast"(%5993, %5994) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %5996 = ttir.empty() : tensor<640x3072xbf16>
    %5997 = "ttir.reshape"(%5995, %5996) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %5998 = ttir.empty() : tensor<1x8192x3072xbf16>
    %5999 = "ttir.reshape"(%arg196, %5998) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %6000 = ttir.empty() : tensor<8192x3072xbf16>
    %6001 = "ttir.reshape"(%5999, %6000) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6002 = ttir.empty() : tensor<3072x8192xbf16>
    %6003 = "ttir.permute"(%6001, %6002) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6004 = "ttir.dot_general"(%5997, %6003) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %6005 = ttir.empty() : tensor<1x640x8192xbf16>
    %6006 = "ttir.reshape"(%6004, %6005) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6007 = ttir.empty() : tensor<1x640x8192xf32>
    %6008 = "ttir.typecast"(%6006, %6007) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6009 = ttir.empty() : tensor<1x640x8192xbf16>
    %6010 = "ttir.sigmoid"(%6006, %6009) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6011 = ttir.empty() : tensor<1x640x8192xf32>
    %6012 = "ttir.typecast"(%6010, %6011) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6013 = ttir.empty() : tensor<1x640x8192xf32>
    %6014 = "ttir.multiply"(%6008, %6012, %6013) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6015 = ttir.empty() : tensor<1x640x8192xbf16>
    %6016 = "ttir.typecast"(%6014, %6015) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6017 = ttir.empty() : tensor<1x640x8192xf32>
    %6018 = "ttir.typecast"(%6016, %6017) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6019 = ttir.empty() : tensor<1x8192x3072xbf16>
    %6020 = "ttir.reshape"(%arg191, %6019) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %6021 = ttir.empty() : tensor<8192x3072xbf16>
    %6022 = "ttir.reshape"(%6020, %6021) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6023 = ttir.empty() : tensor<3072x8192xbf16>
    %6024 = "ttir.permute"(%6022, %6023) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6025 = "ttir.dot_general"(%5997, %6024) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %6026 = ttir.empty() : tensor<1x640x8192xbf16>
    %6027 = "ttir.reshape"(%6025, %6026) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6028 = ttir.empty() : tensor<1x640x8192xf32>
    %6029 = "ttir.typecast"(%6027, %6028) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6030 = ttir.empty() : tensor<1x640x8192xf32>
    %6031 = "ttir.multiply"(%6018, %6029, %6030) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6032 = ttir.empty() : tensor<1x640x8192xbf16>
    %6033 = "ttir.typecast"(%6031, %6032) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6034 = ttir.empty() : tensor<640x8192xbf16>
    %6035 = "ttir.reshape"(%6033, %6034) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %6036 = ttir.empty() : tensor<1x3072x8192xbf16>
    %6037 = "ttir.reshape"(%arg190, %6036) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %6038 = ttir.empty() : tensor<3072x8192xbf16>
    %6039 = "ttir.reshape"(%6037, %6038) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6040 = ttir.empty() : tensor<8192x3072xbf16>
    %6041 = "ttir.permute"(%6039, %6040) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6042 = "ttir.dot_general"(%6035, %6041) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %6043 = ttir.empty() : tensor<1x640x3072xbf16>
    %6044 = "ttir.reshape"(%6042, %6043) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6045 = ttir.empty() : tensor<1x640x3072xbf16>
    %6046 = "ttir.add"(%5955, %6044, %6045) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6047 = ttir.empty() : tensor<1x640x3072xf32>
    %6048 = "ttir.typecast"(%6046, %6047) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6049 = ttir.empty() : tensor<1x640x3072xf32>
    %6050 = "ttir.pow"(%6048, %5, %6049) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6051 = ttir.empty() : tensor<1x640xf32>
    %6052 = "ttir.sum"(%6050, %6051) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6053 = ttir.empty() : tensor<1x640xf32>
    %6054 = "ttir.multiply"(%6052, %4, %6053) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6055 = ttir.empty() : tensor<1x640x1xf32>
    %6056 = "ttir.reshape"(%6054, %6055) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6057 = ttir.empty() : tensor<1x640x1xf32>
    %6058 = "ttir.add"(%6056, %46, %6057) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6059 = ttir.empty() : tensor<1x640x1xf32>
    %6060 = "ttir.rsqrt"(%6058, %6059) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6061 = ttir.empty() : tensor<1x640xf32>
    %6062 = "ttir.reshape"(%6060, %6061) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6063 = ttir.empty() : tensor<1x640x1xf32>
    %6064 = "ttir.reshape"(%6062, %6063) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6065 = ttir.empty() : tensor<1x640x3072xf32>
    %6066 = "ttir.broadcast"(%6064, %6065) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6067 = ttir.empty() : tensor<1x640x3072xf32>
    %6068 = "ttir.multiply"(%6048, %6066, %6067) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6069 = ttir.empty() : tensor<1x640x3072xbf16>
    %6070 = "ttir.typecast"(%6068, %6069) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6071 = ttir.empty() : tensor<1x640x3072xf32>
    %6072 = "ttir.typecast"(%6070, %6071) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6073 = ttir.empty() : tensor<1x640x3072xf32>
    %6074 = "ttir.multiply"(%5818, %6072, %6073) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6075 = ttir.empty() : tensor<1x640x3072xbf16>
    %6076 = "ttir.typecast"(%6074, %6075) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6077 = ttir.empty() : tensor<640x3072xbf16>
    %6078 = "ttir.reshape"(%6076, %6077) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %6079 = ttir.empty() : tensor<1x1024x3072xbf16>
    %6080 = "ttir.reshape"(%arg189, %6079) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %6081 = ttir.empty() : tensor<1024x3072xbf16>
    %6082 = "ttir.reshape"(%6080, %6081) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %6083 = ttir.empty() : tensor<3072x1024xbf16>
    %6084 = "ttir.permute"(%6082, %6083) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %6085 = "ttir.dot_general"(%6078, %6084) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %6086 = ttir.empty() : tensor<1x640x8x128xbf16>
    %6087 = "ttir.reshape"(%6085, %6086) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %6088 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6089 = "ttir.permute"(%6087, %6088) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6090 = ttir.empty() : tensor<1x1x3072xbf16>
    %6091 = "ttir.reshape"(%arg206, %6090) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %6092 = ttir.empty() : tensor<3072xbf16>
    %6093 = "ttir.reshape"(%6091, %6092) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %6094 = ttir.empty() : tensor<3072xf32>
    %6095 = "ttir.typecast"(%6093, %6094) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %6096 = ttir.empty() : tensor<1x1x3072xf32>
    %6097 = "ttir.reshape"(%6095, %6096) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %6098 = ttir.empty() : tensor<1x640x3072xf32>
    %6099 = "ttir.broadcast"(%6097, %6098) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6100 = ttir.empty() : tensor<1x3072x3072xbf16>
    %6101 = "ttir.reshape"(%arg203, %6100) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %6102 = ttir.empty() : tensor<3072x3072xbf16>
    %6103 = "ttir.reshape"(%6101, %6102) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6104 = ttir.empty() : tensor<3072x3072xbf16>
    %6105 = "ttir.permute"(%6103, %6104) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6106 = "ttir.dot_general"(%6078, %6105) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %6107 = ttir.empty() : tensor<1x640x24x128xbf16>
    %6108 = "ttir.reshape"(%6106, %6107) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %6109 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6110 = "ttir.permute"(%6108, %6109) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6111 = ttir.empty() : tensor<1x24x640x128xf32>
    %6112 = "ttir.typecast"(%6110, %6111) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6113 = ttir.empty() : tensor<1x24x640x128xf32>
    %6114 = "ttir.multiply"(%6112, %125, %6113) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6115 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6116 = "ttir.typecast"(%6114, %6115) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6117 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6118 = "ttir.slice_static"(%6110, %6117) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6119 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6120 = "ttir.neg"(%6118, %6119) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6121 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6122 = "ttir.slice_static"(%6110, %6121) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6123 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6124 = "ttir.concat"(%6120, %6122, %6123) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6125 = ttir.empty() : tensor<1x24x640x128xf32>
    %6126 = "ttir.typecast"(%6124, %6125) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6127 = ttir.empty() : tensor<1x24x640x128xf32>
    %6128 = "ttir.multiply"(%6126, %153, %6127) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6129 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6130 = "ttir.typecast"(%6128, %6129) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6131 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6132 = "ttir.add"(%6116, %6130, %6131) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6133 = ttir.empty() : tensor<24x640x128xbf16>
    %6134 = "ttir.reshape"(%6132, %6133) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %6135 = ttir.empty() : tensor<1x1024x3072xbf16>
    %6136 = "ttir.reshape"(%arg202, %6135) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %6137 = ttir.empty() : tensor<1024x3072xbf16>
    %6138 = "ttir.reshape"(%6136, %6137) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %6139 = ttir.empty() : tensor<3072x1024xbf16>
    %6140 = "ttir.permute"(%6138, %6139) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %6141 = "ttir.dot_general"(%6078, %6140) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %6142 = ttir.empty() : tensor<1x640x8x128xbf16>
    %6143 = "ttir.reshape"(%6141, %6142) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %6144 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6145 = "ttir.permute"(%6143, %6144) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6146 = ttir.empty() : tensor<1x8x640x128xf32>
    %6147 = "ttir.typecast"(%6145, %6146) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6148 = ttir.empty() : tensor<1x8x640x128xf32>
    %6149 = "ttir.multiply"(%6147, %178, %6148) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6150 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6151 = "ttir.typecast"(%6149, %6150) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6152 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6153 = "ttir.slice_static"(%6145, %6152) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6154 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6155 = "ttir.neg"(%6153, %6154) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6156 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6157 = "ttir.slice_static"(%6145, %6156) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6158 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6159 = "ttir.concat"(%6155, %6157, %6158) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6160 = ttir.empty() : tensor<1x8x640x128xf32>
    %6161 = "ttir.typecast"(%6159, %6160) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6162 = ttir.empty() : tensor<1x8x640x128xf32>
    %6163 = "ttir.multiply"(%6161, %196, %6162) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6164 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6165 = "ttir.typecast"(%6163, %6164) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6166 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6167 = "ttir.add"(%6151, %6165, %6166) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6168 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %6169 = "ttir.reshape"(%6167, %6168) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %6170 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %6171 = "ttir.broadcast"(%6169, %6170) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %6172 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6173 = "ttir.reshape"(%6171, %6172) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6174 = ttir.empty() : tensor<1x24x128x640xbf16>
    %6175 = "ttir.permute"(%6173, %6174) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %6176 = ttir.empty() : tensor<24x128x640xbf16>
    %6177 = "ttir.reshape"(%6175, %6176) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %6178 = "ttir.dot_general"(%6134, %6177) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %6179 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6180 = "ttir.reshape"(%6178, %6179) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6181 = ttir.empty() : tensor<1x24x640x640xf32>
    %6182 = "ttir.typecast"(%6180, %6181) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6183 = ttir.empty() : tensor<1x24x640x640xf32>
    %6184 = "ttir.multiply"(%6182, %221, %6183) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6185 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6186 = "ttir.typecast"(%6184, %6185) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6187 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6188 = "ttir.add"(%6186, %285, %6187) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6189 = ttir.empty() : tensor<1x24x640x640xf32>
    %6190 = "ttir.typecast"(%6188, %6189) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6191 = ttir.empty() : tensor<1x24x640xf32>
    %6192 = "ttir.max"(%6190, %6191) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %6193 = ttir.empty() : tensor<1x24x640x1xf32>
    %6194 = "ttir.reshape"(%6192, %6193) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %6195 = ttir.empty() : tensor<1x24x640x640xf32>
    %6196 = "ttir.broadcast"(%6194, %6195) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6197 = ttir.empty() : tensor<1x24x640x640xf32>
    %6198 = "ttir.subtract"(%6190, %6196, %6197) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6199 = ttir.empty() : tensor<1x24x640x640xf32>
    %6200 = "ttir.exp"(%6198, %6199) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6201 = ttir.empty() : tensor<1x24x640xf32>
    %6202 = "ttir.sum"(%6200, %6201) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %6203 = ttir.empty() : tensor<1x24x640x1xf32>
    %6204 = "ttir.reshape"(%6202, %6203) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %6205 = ttir.empty() : tensor<1x24x640x640xf32>
    %6206 = "ttir.broadcast"(%6204, %6205) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6207 = ttir.empty() : tensor<1x24x640x640xf32>
    %6208 = "ttir.div"(%6200, %6206, %6207) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6209 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6210 = "ttir.typecast"(%6208, %6209) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6211 = ttir.empty() : tensor<24x640x640xbf16>
    %6212 = "ttir.reshape"(%6210, %6211) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %6213 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %6214 = "ttir.reshape"(%6089, %6213) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %6215 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %6216 = "ttir.broadcast"(%6214, %6215) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %6217 = ttir.empty() : tensor<24x640x128xbf16>
    %6218 = "ttir.reshape"(%6216, %6217) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %6219 = "ttir.dot_general"(%6212, %6218) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %6220 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6221 = "ttir.reshape"(%6219, %6220) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6222 = ttir.empty() : tensor<1x640x24x128xbf16>
    %6223 = "ttir.permute"(%6221, %6222) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %6224 = ttir.empty() : tensor<640x3072xbf16>
    %6225 = "ttir.reshape"(%6223, %6224) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %6226 = ttir.empty() : tensor<1x3072x3072xbf16>
    %6227 = "ttir.reshape"(%arg201, %6226) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %6228 = ttir.empty() : tensor<3072x3072xbf16>
    %6229 = "ttir.reshape"(%6227, %6228) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6230 = ttir.empty() : tensor<3072x3072xbf16>
    %6231 = "ttir.permute"(%6229, %6230) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6232 = "ttir.dot_general"(%6225, %6231) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %6233 = ttir.empty() : tensor<1x640x3072xbf16>
    %6234 = "ttir.reshape"(%6232, %6233) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6235 = ttir.empty() : tensor<1x640x3072xbf16>
    %6236 = "ttir.add"(%6046, %6234, %6235) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6237 = ttir.empty() : tensor<1x1x3072xbf16>
    %6238 = "ttir.reshape"(%arg204, %6237) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %6239 = ttir.empty() : tensor<3072xbf16>
    %6240 = "ttir.reshape"(%6238, %6239) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %6241 = ttir.empty() : tensor<3072xf32>
    %6242 = "ttir.typecast"(%6240, %6241) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %6243 = ttir.empty() : tensor<1x1x3072xf32>
    %6244 = "ttir.reshape"(%6242, %6243) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %6245 = ttir.empty() : tensor<1x640x3072xf32>
    %6246 = "ttir.broadcast"(%6244, %6245) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6247 = ttir.empty() : tensor<1x640x3072xf32>
    %6248 = "ttir.typecast"(%6236, %6247) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6249 = ttir.empty() : tensor<1x640x3072xf32>
    %6250 = "ttir.pow"(%6248, %5, %6249) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6251 = ttir.empty() : tensor<1x640xf32>
    %6252 = "ttir.sum"(%6250, %6251) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6253 = ttir.empty() : tensor<1x640xf32>
    %6254 = "ttir.multiply"(%6252, %4, %6253) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6255 = ttir.empty() : tensor<1x640x1xf32>
    %6256 = "ttir.reshape"(%6254, %6255) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6257 = ttir.empty() : tensor<1x640x1xf32>
    %6258 = "ttir.add"(%6256, %46, %6257) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6259 = ttir.empty() : tensor<1x640x1xf32>
    %6260 = "ttir.rsqrt"(%6258, %6259) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6261 = ttir.empty() : tensor<1x640xf32>
    %6262 = "ttir.reshape"(%6260, %6261) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6263 = ttir.empty() : tensor<1x640x1xf32>
    %6264 = "ttir.reshape"(%6262, %6263) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6265 = ttir.empty() : tensor<1x640x3072xf32>
    %6266 = "ttir.broadcast"(%6264, %6265) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6267 = ttir.empty() : tensor<1x640x3072xf32>
    %6268 = "ttir.multiply"(%6248, %6266, %6267) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6269 = ttir.empty() : tensor<1x640x3072xbf16>
    %6270 = "ttir.typecast"(%6268, %6269) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6271 = ttir.empty() : tensor<1x640x3072xf32>
    %6272 = "ttir.typecast"(%6270, %6271) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6273 = ttir.empty() : tensor<1x640x3072xf32>
    %6274 = "ttir.multiply"(%6246, %6272, %6273) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6275 = ttir.empty() : tensor<1x640x3072xbf16>
    %6276 = "ttir.typecast"(%6274, %6275) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6277 = ttir.empty() : tensor<640x3072xbf16>
    %6278 = "ttir.reshape"(%6276, %6277) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %6279 = ttir.empty() : tensor<1x8192x3072xbf16>
    %6280 = "ttir.reshape"(%arg205, %6279) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %6281 = ttir.empty() : tensor<8192x3072xbf16>
    %6282 = "ttir.reshape"(%6280, %6281) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6283 = ttir.empty() : tensor<3072x8192xbf16>
    %6284 = "ttir.permute"(%6282, %6283) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6285 = "ttir.dot_general"(%6278, %6284) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %6286 = ttir.empty() : tensor<1x640x8192xbf16>
    %6287 = "ttir.reshape"(%6285, %6286) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6288 = ttir.empty() : tensor<1x640x8192xf32>
    %6289 = "ttir.typecast"(%6287, %6288) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6290 = ttir.empty() : tensor<1x640x8192xbf16>
    %6291 = "ttir.sigmoid"(%6287, %6290) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6292 = ttir.empty() : tensor<1x640x8192xf32>
    %6293 = "ttir.typecast"(%6291, %6292) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6294 = ttir.empty() : tensor<1x640x8192xf32>
    %6295 = "ttir.multiply"(%6289, %6293, %6294) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6296 = ttir.empty() : tensor<1x640x8192xbf16>
    %6297 = "ttir.typecast"(%6295, %6296) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6298 = ttir.empty() : tensor<1x640x8192xf32>
    %6299 = "ttir.typecast"(%6297, %6298) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6300 = ttir.empty() : tensor<1x8192x3072xbf16>
    %6301 = "ttir.reshape"(%arg200, %6300) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %6302 = ttir.empty() : tensor<8192x3072xbf16>
    %6303 = "ttir.reshape"(%6301, %6302) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6304 = ttir.empty() : tensor<3072x8192xbf16>
    %6305 = "ttir.permute"(%6303, %6304) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6306 = "ttir.dot_general"(%6278, %6305) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %6307 = ttir.empty() : tensor<1x640x8192xbf16>
    %6308 = "ttir.reshape"(%6306, %6307) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6309 = ttir.empty() : tensor<1x640x8192xf32>
    %6310 = "ttir.typecast"(%6308, %6309) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6311 = ttir.empty() : tensor<1x640x8192xf32>
    %6312 = "ttir.multiply"(%6299, %6310, %6311) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6313 = ttir.empty() : tensor<1x640x8192xbf16>
    %6314 = "ttir.typecast"(%6312, %6313) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6315 = ttir.empty() : tensor<640x8192xbf16>
    %6316 = "ttir.reshape"(%6314, %6315) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %6317 = ttir.empty() : tensor<1x3072x8192xbf16>
    %6318 = "ttir.reshape"(%arg199, %6317) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %6319 = ttir.empty() : tensor<3072x8192xbf16>
    %6320 = "ttir.reshape"(%6318, %6319) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6321 = ttir.empty() : tensor<8192x3072xbf16>
    %6322 = "ttir.permute"(%6320, %6321) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6323 = "ttir.dot_general"(%6316, %6322) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %6324 = ttir.empty() : tensor<1x640x3072xbf16>
    %6325 = "ttir.reshape"(%6323, %6324) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6326 = ttir.empty() : tensor<1x640x3072xbf16>
    %6327 = "ttir.add"(%6236, %6325, %6326) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6328 = ttir.empty() : tensor<1x640x3072xf32>
    %6329 = "ttir.typecast"(%6327, %6328) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6330 = ttir.empty() : tensor<1x640x3072xf32>
    %6331 = "ttir.pow"(%6329, %5, %6330) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6332 = ttir.empty() : tensor<1x640xf32>
    %6333 = "ttir.sum"(%6331, %6332) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6334 = ttir.empty() : tensor<1x640xf32>
    %6335 = "ttir.multiply"(%6333, %4, %6334) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6336 = ttir.empty() : tensor<1x640x1xf32>
    %6337 = "ttir.reshape"(%6335, %6336) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6338 = ttir.empty() : tensor<1x640x1xf32>
    %6339 = "ttir.add"(%6337, %46, %6338) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6340 = ttir.empty() : tensor<1x640x1xf32>
    %6341 = "ttir.rsqrt"(%6339, %6340) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6342 = ttir.empty() : tensor<1x640xf32>
    %6343 = "ttir.reshape"(%6341, %6342) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6344 = ttir.empty() : tensor<1x640x1xf32>
    %6345 = "ttir.reshape"(%6343, %6344) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6346 = ttir.empty() : tensor<1x640x3072xf32>
    %6347 = "ttir.broadcast"(%6345, %6346) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6348 = ttir.empty() : tensor<1x640x3072xf32>
    %6349 = "ttir.multiply"(%6329, %6347, %6348) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6350 = ttir.empty() : tensor<1x640x3072xbf16>
    %6351 = "ttir.typecast"(%6349, %6350) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6352 = ttir.empty() : tensor<1x640x3072xf32>
    %6353 = "ttir.typecast"(%6351, %6352) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6354 = ttir.empty() : tensor<1x640x3072xf32>
    %6355 = "ttir.multiply"(%6099, %6353, %6354) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6356 = ttir.empty() : tensor<1x640x3072xbf16>
    %6357 = "ttir.typecast"(%6355, %6356) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6358 = ttir.empty() : tensor<640x3072xbf16>
    %6359 = "ttir.reshape"(%6357, %6358) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %6360 = ttir.empty() : tensor<1x1024x3072xbf16>
    %6361 = "ttir.reshape"(%arg198, %6360) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %6362 = ttir.empty() : tensor<1024x3072xbf16>
    %6363 = "ttir.reshape"(%6361, %6362) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %6364 = ttir.empty() : tensor<3072x1024xbf16>
    %6365 = "ttir.permute"(%6363, %6364) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %6366 = "ttir.dot_general"(%6359, %6365) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %6367 = ttir.empty() : tensor<1x640x8x128xbf16>
    %6368 = "ttir.reshape"(%6366, %6367) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %6369 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6370 = "ttir.permute"(%6368, %6369) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6371 = ttir.empty() : tensor<1x1x3072xbf16>
    %6372 = "ttir.reshape"(%arg215, %6371) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %6373 = ttir.empty() : tensor<3072xbf16>
    %6374 = "ttir.reshape"(%6372, %6373) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %6375 = ttir.empty() : tensor<3072xf32>
    %6376 = "ttir.typecast"(%6374, %6375) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %6377 = ttir.empty() : tensor<1x1x3072xf32>
    %6378 = "ttir.reshape"(%6376, %6377) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %6379 = ttir.empty() : tensor<1x640x3072xf32>
    %6380 = "ttir.broadcast"(%6378, %6379) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6381 = ttir.empty() : tensor<1x3072x3072xbf16>
    %6382 = "ttir.reshape"(%arg212, %6381) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %6383 = ttir.empty() : tensor<3072x3072xbf16>
    %6384 = "ttir.reshape"(%6382, %6383) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6385 = ttir.empty() : tensor<3072x3072xbf16>
    %6386 = "ttir.permute"(%6384, %6385) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6387 = "ttir.dot_general"(%6359, %6386) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %6388 = ttir.empty() : tensor<1x640x24x128xbf16>
    %6389 = "ttir.reshape"(%6387, %6388) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %6390 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6391 = "ttir.permute"(%6389, %6390) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6392 = ttir.empty() : tensor<1x24x640x128xf32>
    %6393 = "ttir.typecast"(%6391, %6392) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6394 = ttir.empty() : tensor<1x24x640x128xf32>
    %6395 = "ttir.multiply"(%6393, %125, %6394) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6396 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6397 = "ttir.typecast"(%6395, %6396) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6398 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6399 = "ttir.slice_static"(%6391, %6398) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6400 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6401 = "ttir.neg"(%6399, %6400) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6402 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6403 = "ttir.slice_static"(%6391, %6402) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6404 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6405 = "ttir.concat"(%6401, %6403, %6404) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6406 = ttir.empty() : tensor<1x24x640x128xf32>
    %6407 = "ttir.typecast"(%6405, %6406) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6408 = ttir.empty() : tensor<1x24x640x128xf32>
    %6409 = "ttir.multiply"(%6407, %153, %6408) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6410 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6411 = "ttir.typecast"(%6409, %6410) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6412 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6413 = "ttir.add"(%6397, %6411, %6412) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6414 = ttir.empty() : tensor<24x640x128xbf16>
    %6415 = "ttir.reshape"(%6413, %6414) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %6416 = ttir.empty() : tensor<1x1024x3072xbf16>
    %6417 = "ttir.reshape"(%arg211, %6416) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %6418 = ttir.empty() : tensor<1024x3072xbf16>
    %6419 = "ttir.reshape"(%6417, %6418) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %6420 = ttir.empty() : tensor<3072x1024xbf16>
    %6421 = "ttir.permute"(%6419, %6420) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %6422 = "ttir.dot_general"(%6359, %6421) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %6423 = ttir.empty() : tensor<1x640x8x128xbf16>
    %6424 = "ttir.reshape"(%6422, %6423) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %6425 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6426 = "ttir.permute"(%6424, %6425) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6427 = ttir.empty() : tensor<1x8x640x128xf32>
    %6428 = "ttir.typecast"(%6426, %6427) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6429 = ttir.empty() : tensor<1x8x640x128xf32>
    %6430 = "ttir.multiply"(%6428, %178, %6429) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6431 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6432 = "ttir.typecast"(%6430, %6431) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6433 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6434 = "ttir.slice_static"(%6426, %6433) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6435 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6436 = "ttir.neg"(%6434, %6435) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6437 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6438 = "ttir.slice_static"(%6426, %6437) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6439 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6440 = "ttir.concat"(%6436, %6438, %6439) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6441 = ttir.empty() : tensor<1x8x640x128xf32>
    %6442 = "ttir.typecast"(%6440, %6441) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6443 = ttir.empty() : tensor<1x8x640x128xf32>
    %6444 = "ttir.multiply"(%6442, %196, %6443) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6445 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6446 = "ttir.typecast"(%6444, %6445) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6447 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6448 = "ttir.add"(%6432, %6446, %6447) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6449 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %6450 = "ttir.reshape"(%6448, %6449) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %6451 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %6452 = "ttir.broadcast"(%6450, %6451) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %6453 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6454 = "ttir.reshape"(%6452, %6453) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6455 = ttir.empty() : tensor<1x24x128x640xbf16>
    %6456 = "ttir.permute"(%6454, %6455) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %6457 = ttir.empty() : tensor<24x128x640xbf16>
    %6458 = "ttir.reshape"(%6456, %6457) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %6459 = "ttir.dot_general"(%6415, %6458) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %6460 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6461 = "ttir.reshape"(%6459, %6460) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6462 = ttir.empty() : tensor<1x24x640x640xf32>
    %6463 = "ttir.typecast"(%6461, %6462) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6464 = ttir.empty() : tensor<1x24x640x640xf32>
    %6465 = "ttir.multiply"(%6463, %221, %6464) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6466 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6467 = "ttir.typecast"(%6465, %6466) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6468 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6469 = "ttir.add"(%6467, %285, %6468) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6470 = ttir.empty() : tensor<1x24x640x640xf32>
    %6471 = "ttir.typecast"(%6469, %6470) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6472 = ttir.empty() : tensor<1x24x640xf32>
    %6473 = "ttir.max"(%6471, %6472) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %6474 = ttir.empty() : tensor<1x24x640x1xf32>
    %6475 = "ttir.reshape"(%6473, %6474) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %6476 = ttir.empty() : tensor<1x24x640x640xf32>
    %6477 = "ttir.broadcast"(%6475, %6476) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6478 = ttir.empty() : tensor<1x24x640x640xf32>
    %6479 = "ttir.subtract"(%6471, %6477, %6478) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6480 = ttir.empty() : tensor<1x24x640x640xf32>
    %6481 = "ttir.exp"(%6479, %6480) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6482 = ttir.empty() : tensor<1x24x640xf32>
    %6483 = "ttir.sum"(%6481, %6482) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %6484 = ttir.empty() : tensor<1x24x640x1xf32>
    %6485 = "ttir.reshape"(%6483, %6484) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %6486 = ttir.empty() : tensor<1x24x640x640xf32>
    %6487 = "ttir.broadcast"(%6485, %6486) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6488 = ttir.empty() : tensor<1x24x640x640xf32>
    %6489 = "ttir.div"(%6481, %6487, %6488) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6490 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6491 = "ttir.typecast"(%6489, %6490) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6492 = ttir.empty() : tensor<24x640x640xbf16>
    %6493 = "ttir.reshape"(%6491, %6492) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %6494 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %6495 = "ttir.reshape"(%6370, %6494) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %6496 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %6497 = "ttir.broadcast"(%6495, %6496) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %6498 = ttir.empty() : tensor<24x640x128xbf16>
    %6499 = "ttir.reshape"(%6497, %6498) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %6500 = "ttir.dot_general"(%6493, %6499) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %6501 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6502 = "ttir.reshape"(%6500, %6501) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6503 = ttir.empty() : tensor<1x640x24x128xbf16>
    %6504 = "ttir.permute"(%6502, %6503) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %6505 = ttir.empty() : tensor<640x3072xbf16>
    %6506 = "ttir.reshape"(%6504, %6505) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %6507 = ttir.empty() : tensor<1x3072x3072xbf16>
    %6508 = "ttir.reshape"(%arg210, %6507) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %6509 = ttir.empty() : tensor<3072x3072xbf16>
    %6510 = "ttir.reshape"(%6508, %6509) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6511 = ttir.empty() : tensor<3072x3072xbf16>
    %6512 = "ttir.permute"(%6510, %6511) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6513 = "ttir.dot_general"(%6506, %6512) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %6514 = ttir.empty() : tensor<1x640x3072xbf16>
    %6515 = "ttir.reshape"(%6513, %6514) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6516 = ttir.empty() : tensor<1x640x3072xbf16>
    %6517 = "ttir.add"(%6327, %6515, %6516) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6518 = ttir.empty() : tensor<1x1x3072xbf16>
    %6519 = "ttir.reshape"(%arg213, %6518) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %6520 = ttir.empty() : tensor<3072xbf16>
    %6521 = "ttir.reshape"(%6519, %6520) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %6522 = ttir.empty() : tensor<3072xf32>
    %6523 = "ttir.typecast"(%6521, %6522) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %6524 = ttir.empty() : tensor<1x1x3072xf32>
    %6525 = "ttir.reshape"(%6523, %6524) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %6526 = ttir.empty() : tensor<1x640x3072xf32>
    %6527 = "ttir.broadcast"(%6525, %6526) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6528 = ttir.empty() : tensor<1x640x3072xf32>
    %6529 = "ttir.typecast"(%6517, %6528) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6530 = ttir.empty() : tensor<1x640x3072xf32>
    %6531 = "ttir.pow"(%6529, %5, %6530) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6532 = ttir.empty() : tensor<1x640xf32>
    %6533 = "ttir.sum"(%6531, %6532) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6534 = ttir.empty() : tensor<1x640xf32>
    %6535 = "ttir.multiply"(%6533, %4, %6534) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6536 = ttir.empty() : tensor<1x640x1xf32>
    %6537 = "ttir.reshape"(%6535, %6536) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6538 = ttir.empty() : tensor<1x640x1xf32>
    %6539 = "ttir.add"(%6537, %46, %6538) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6540 = ttir.empty() : tensor<1x640x1xf32>
    %6541 = "ttir.rsqrt"(%6539, %6540) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6542 = ttir.empty() : tensor<1x640xf32>
    %6543 = "ttir.reshape"(%6541, %6542) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6544 = ttir.empty() : tensor<1x640x1xf32>
    %6545 = "ttir.reshape"(%6543, %6544) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6546 = ttir.empty() : tensor<1x640x3072xf32>
    %6547 = "ttir.broadcast"(%6545, %6546) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6548 = ttir.empty() : tensor<1x640x3072xf32>
    %6549 = "ttir.multiply"(%6529, %6547, %6548) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6550 = ttir.empty() : tensor<1x640x3072xbf16>
    %6551 = "ttir.typecast"(%6549, %6550) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6552 = ttir.empty() : tensor<1x640x3072xf32>
    %6553 = "ttir.typecast"(%6551, %6552) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6554 = ttir.empty() : tensor<1x640x3072xf32>
    %6555 = "ttir.multiply"(%6527, %6553, %6554) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6556 = ttir.empty() : tensor<1x640x3072xbf16>
    %6557 = "ttir.typecast"(%6555, %6556) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6558 = ttir.empty() : tensor<640x3072xbf16>
    %6559 = "ttir.reshape"(%6557, %6558) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %6560 = ttir.empty() : tensor<1x8192x3072xbf16>
    %6561 = "ttir.reshape"(%arg214, %6560) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %6562 = ttir.empty() : tensor<8192x3072xbf16>
    %6563 = "ttir.reshape"(%6561, %6562) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6564 = ttir.empty() : tensor<3072x8192xbf16>
    %6565 = "ttir.permute"(%6563, %6564) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6566 = "ttir.dot_general"(%6559, %6565) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %6567 = ttir.empty() : tensor<1x640x8192xbf16>
    %6568 = "ttir.reshape"(%6566, %6567) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6569 = ttir.empty() : tensor<1x640x8192xf32>
    %6570 = "ttir.typecast"(%6568, %6569) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6571 = ttir.empty() : tensor<1x640x8192xbf16>
    %6572 = "ttir.sigmoid"(%6568, %6571) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6573 = ttir.empty() : tensor<1x640x8192xf32>
    %6574 = "ttir.typecast"(%6572, %6573) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6575 = ttir.empty() : tensor<1x640x8192xf32>
    %6576 = "ttir.multiply"(%6570, %6574, %6575) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6577 = ttir.empty() : tensor<1x640x8192xbf16>
    %6578 = "ttir.typecast"(%6576, %6577) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6579 = ttir.empty() : tensor<1x640x8192xf32>
    %6580 = "ttir.typecast"(%6578, %6579) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6581 = ttir.empty() : tensor<1x8192x3072xbf16>
    %6582 = "ttir.reshape"(%arg209, %6581) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %6583 = ttir.empty() : tensor<8192x3072xbf16>
    %6584 = "ttir.reshape"(%6582, %6583) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6585 = ttir.empty() : tensor<3072x8192xbf16>
    %6586 = "ttir.permute"(%6584, %6585) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6587 = "ttir.dot_general"(%6559, %6586) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %6588 = ttir.empty() : tensor<1x640x8192xbf16>
    %6589 = "ttir.reshape"(%6587, %6588) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6590 = ttir.empty() : tensor<1x640x8192xf32>
    %6591 = "ttir.typecast"(%6589, %6590) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6592 = ttir.empty() : tensor<1x640x8192xf32>
    %6593 = "ttir.multiply"(%6580, %6591, %6592) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6594 = ttir.empty() : tensor<1x640x8192xbf16>
    %6595 = "ttir.typecast"(%6593, %6594) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6596 = ttir.empty() : tensor<640x8192xbf16>
    %6597 = "ttir.reshape"(%6595, %6596) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %6598 = ttir.empty() : tensor<1x3072x8192xbf16>
    %6599 = "ttir.reshape"(%arg208, %6598) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %6600 = ttir.empty() : tensor<3072x8192xbf16>
    %6601 = "ttir.reshape"(%6599, %6600) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6602 = ttir.empty() : tensor<8192x3072xbf16>
    %6603 = "ttir.permute"(%6601, %6602) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6604 = "ttir.dot_general"(%6597, %6603) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %6605 = ttir.empty() : tensor<1x640x3072xbf16>
    %6606 = "ttir.reshape"(%6604, %6605) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6607 = ttir.empty() : tensor<1x640x3072xbf16>
    %6608 = "ttir.add"(%6517, %6606, %6607) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6609 = ttir.empty() : tensor<1x640x3072xf32>
    %6610 = "ttir.typecast"(%6608, %6609) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6611 = ttir.empty() : tensor<1x640x3072xf32>
    %6612 = "ttir.pow"(%6610, %5, %6611) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6613 = ttir.empty() : tensor<1x640xf32>
    %6614 = "ttir.sum"(%6612, %6613) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6615 = ttir.empty() : tensor<1x640xf32>
    %6616 = "ttir.multiply"(%6614, %4, %6615) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6617 = ttir.empty() : tensor<1x640x1xf32>
    %6618 = "ttir.reshape"(%6616, %6617) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6619 = ttir.empty() : tensor<1x640x1xf32>
    %6620 = "ttir.add"(%6618, %46, %6619) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6621 = ttir.empty() : tensor<1x640x1xf32>
    %6622 = "ttir.rsqrt"(%6620, %6621) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6623 = ttir.empty() : tensor<1x640xf32>
    %6624 = "ttir.reshape"(%6622, %6623) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6625 = ttir.empty() : tensor<1x640x1xf32>
    %6626 = "ttir.reshape"(%6624, %6625) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6627 = ttir.empty() : tensor<1x640x3072xf32>
    %6628 = "ttir.broadcast"(%6626, %6627) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6629 = ttir.empty() : tensor<1x640x3072xf32>
    %6630 = "ttir.multiply"(%6610, %6628, %6629) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6631 = ttir.empty() : tensor<1x640x3072xbf16>
    %6632 = "ttir.typecast"(%6630, %6631) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6633 = ttir.empty() : tensor<1x640x3072xf32>
    %6634 = "ttir.typecast"(%6632, %6633) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6635 = ttir.empty() : tensor<1x640x3072xf32>
    %6636 = "ttir.multiply"(%6380, %6634, %6635) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6637 = ttir.empty() : tensor<1x640x3072xbf16>
    %6638 = "ttir.typecast"(%6636, %6637) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6639 = ttir.empty() : tensor<640x3072xbf16>
    %6640 = "ttir.reshape"(%6638, %6639) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %6641 = ttir.empty() : tensor<1x1024x3072xbf16>
    %6642 = "ttir.reshape"(%arg207, %6641) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %6643 = ttir.empty() : tensor<1024x3072xbf16>
    %6644 = "ttir.reshape"(%6642, %6643) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %6645 = ttir.empty() : tensor<3072x1024xbf16>
    %6646 = "ttir.permute"(%6644, %6645) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %6647 = "ttir.dot_general"(%6640, %6646) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %6648 = ttir.empty() : tensor<1x640x8x128xbf16>
    %6649 = "ttir.reshape"(%6647, %6648) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %6650 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6651 = "ttir.permute"(%6649, %6650) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6652 = ttir.empty() : tensor<1x1x3072xbf16>
    %6653 = "ttir.reshape"(%arg224, %6652) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %6654 = ttir.empty() : tensor<3072xbf16>
    %6655 = "ttir.reshape"(%6653, %6654) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %6656 = ttir.empty() : tensor<3072xf32>
    %6657 = "ttir.typecast"(%6655, %6656) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %6658 = ttir.empty() : tensor<1x1x3072xf32>
    %6659 = "ttir.reshape"(%6657, %6658) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %6660 = ttir.empty() : tensor<1x640x3072xf32>
    %6661 = "ttir.broadcast"(%6659, %6660) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6662 = ttir.empty() : tensor<1x3072x3072xbf16>
    %6663 = "ttir.reshape"(%arg221, %6662) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %6664 = ttir.empty() : tensor<3072x3072xbf16>
    %6665 = "ttir.reshape"(%6663, %6664) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6666 = ttir.empty() : tensor<3072x3072xbf16>
    %6667 = "ttir.permute"(%6665, %6666) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6668 = "ttir.dot_general"(%6640, %6667) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %6669 = ttir.empty() : tensor<1x640x24x128xbf16>
    %6670 = "ttir.reshape"(%6668, %6669) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %6671 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6672 = "ttir.permute"(%6670, %6671) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6673 = ttir.empty() : tensor<1x24x640x128xf32>
    %6674 = "ttir.typecast"(%6672, %6673) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6675 = ttir.empty() : tensor<1x24x640x128xf32>
    %6676 = "ttir.multiply"(%6674, %125, %6675) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6677 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6678 = "ttir.typecast"(%6676, %6677) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6679 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6680 = "ttir.slice_static"(%6672, %6679) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6681 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6682 = "ttir.neg"(%6680, %6681) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6683 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6684 = "ttir.slice_static"(%6672, %6683) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6685 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6686 = "ttir.concat"(%6682, %6684, %6685) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6687 = ttir.empty() : tensor<1x24x640x128xf32>
    %6688 = "ttir.typecast"(%6686, %6687) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6689 = ttir.empty() : tensor<1x24x640x128xf32>
    %6690 = "ttir.multiply"(%6688, %153, %6689) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6691 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6692 = "ttir.typecast"(%6690, %6691) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6693 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6694 = "ttir.add"(%6678, %6692, %6693) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6695 = ttir.empty() : tensor<24x640x128xbf16>
    %6696 = "ttir.reshape"(%6694, %6695) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %6697 = ttir.empty() : tensor<1x1024x3072xbf16>
    %6698 = "ttir.reshape"(%arg220, %6697) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %6699 = ttir.empty() : tensor<1024x3072xbf16>
    %6700 = "ttir.reshape"(%6698, %6699) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %6701 = ttir.empty() : tensor<3072x1024xbf16>
    %6702 = "ttir.permute"(%6700, %6701) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %6703 = "ttir.dot_general"(%6640, %6702) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %6704 = ttir.empty() : tensor<1x640x8x128xbf16>
    %6705 = "ttir.reshape"(%6703, %6704) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %6706 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6707 = "ttir.permute"(%6705, %6706) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6708 = ttir.empty() : tensor<1x8x640x128xf32>
    %6709 = "ttir.typecast"(%6707, %6708) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6710 = ttir.empty() : tensor<1x8x640x128xf32>
    %6711 = "ttir.multiply"(%6709, %178, %6710) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6712 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6713 = "ttir.typecast"(%6711, %6712) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6714 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6715 = "ttir.slice_static"(%6707, %6714) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6716 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6717 = "ttir.neg"(%6715, %6716) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6718 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6719 = "ttir.slice_static"(%6707, %6718) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6720 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6721 = "ttir.concat"(%6717, %6719, %6720) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6722 = ttir.empty() : tensor<1x8x640x128xf32>
    %6723 = "ttir.typecast"(%6721, %6722) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6724 = ttir.empty() : tensor<1x8x640x128xf32>
    %6725 = "ttir.multiply"(%6723, %196, %6724) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6726 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6727 = "ttir.typecast"(%6725, %6726) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6728 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6729 = "ttir.add"(%6713, %6727, %6728) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6730 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %6731 = "ttir.reshape"(%6729, %6730) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %6732 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %6733 = "ttir.broadcast"(%6731, %6732) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %6734 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6735 = "ttir.reshape"(%6733, %6734) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6736 = ttir.empty() : tensor<1x24x128x640xbf16>
    %6737 = "ttir.permute"(%6735, %6736) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %6738 = ttir.empty() : tensor<24x128x640xbf16>
    %6739 = "ttir.reshape"(%6737, %6738) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %6740 = "ttir.dot_general"(%6696, %6739) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %6741 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6742 = "ttir.reshape"(%6740, %6741) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6743 = ttir.empty() : tensor<1x24x640x640xf32>
    %6744 = "ttir.typecast"(%6742, %6743) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6745 = ttir.empty() : tensor<1x24x640x640xf32>
    %6746 = "ttir.multiply"(%6744, %221, %6745) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6747 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6748 = "ttir.typecast"(%6746, %6747) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6749 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6750 = "ttir.add"(%6748, %285, %6749) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6751 = ttir.empty() : tensor<1x24x640x640xf32>
    %6752 = "ttir.typecast"(%6750, %6751) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6753 = ttir.empty() : tensor<1x24x640xf32>
    %6754 = "ttir.max"(%6752, %6753) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %6755 = ttir.empty() : tensor<1x24x640x1xf32>
    %6756 = "ttir.reshape"(%6754, %6755) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %6757 = ttir.empty() : tensor<1x24x640x640xf32>
    %6758 = "ttir.broadcast"(%6756, %6757) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6759 = ttir.empty() : tensor<1x24x640x640xf32>
    %6760 = "ttir.subtract"(%6752, %6758, %6759) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6761 = ttir.empty() : tensor<1x24x640x640xf32>
    %6762 = "ttir.exp"(%6760, %6761) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6763 = ttir.empty() : tensor<1x24x640xf32>
    %6764 = "ttir.sum"(%6762, %6763) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %6765 = ttir.empty() : tensor<1x24x640x1xf32>
    %6766 = "ttir.reshape"(%6764, %6765) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %6767 = ttir.empty() : tensor<1x24x640x640xf32>
    %6768 = "ttir.broadcast"(%6766, %6767) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6769 = ttir.empty() : tensor<1x24x640x640xf32>
    %6770 = "ttir.div"(%6762, %6768, %6769) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %6771 = ttir.empty() : tensor<1x24x640x640xbf16>
    %6772 = "ttir.typecast"(%6770, %6771) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %6773 = ttir.empty() : tensor<24x640x640xbf16>
    %6774 = "ttir.reshape"(%6772, %6773) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %6775 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %6776 = "ttir.reshape"(%6651, %6775) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %6777 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %6778 = "ttir.broadcast"(%6776, %6777) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %6779 = ttir.empty() : tensor<24x640x128xbf16>
    %6780 = "ttir.reshape"(%6778, %6779) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %6781 = "ttir.dot_general"(%6774, %6780) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %6782 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6783 = "ttir.reshape"(%6781, %6782) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6784 = ttir.empty() : tensor<1x640x24x128xbf16>
    %6785 = "ttir.permute"(%6783, %6784) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %6786 = ttir.empty() : tensor<640x3072xbf16>
    %6787 = "ttir.reshape"(%6785, %6786) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %6788 = ttir.empty() : tensor<1x3072x3072xbf16>
    %6789 = "ttir.reshape"(%arg219, %6788) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %6790 = ttir.empty() : tensor<3072x3072xbf16>
    %6791 = "ttir.reshape"(%6789, %6790) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6792 = ttir.empty() : tensor<3072x3072xbf16>
    %6793 = "ttir.permute"(%6791, %6792) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6794 = "ttir.dot_general"(%6787, %6793) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %6795 = ttir.empty() : tensor<1x640x3072xbf16>
    %6796 = "ttir.reshape"(%6794, %6795) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6797 = ttir.empty() : tensor<1x640x3072xbf16>
    %6798 = "ttir.add"(%6608, %6796, %6797) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6799 = ttir.empty() : tensor<1x1x3072xbf16>
    %6800 = "ttir.reshape"(%arg222, %6799) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %6801 = ttir.empty() : tensor<3072xbf16>
    %6802 = "ttir.reshape"(%6800, %6801) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %6803 = ttir.empty() : tensor<3072xf32>
    %6804 = "ttir.typecast"(%6802, %6803) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %6805 = ttir.empty() : tensor<1x1x3072xf32>
    %6806 = "ttir.reshape"(%6804, %6805) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %6807 = ttir.empty() : tensor<1x640x3072xf32>
    %6808 = "ttir.broadcast"(%6806, %6807) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6809 = ttir.empty() : tensor<1x640x3072xf32>
    %6810 = "ttir.typecast"(%6798, %6809) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6811 = ttir.empty() : tensor<1x640x3072xf32>
    %6812 = "ttir.pow"(%6810, %5, %6811) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6813 = ttir.empty() : tensor<1x640xf32>
    %6814 = "ttir.sum"(%6812, %6813) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6815 = ttir.empty() : tensor<1x640xf32>
    %6816 = "ttir.multiply"(%6814, %4, %6815) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6817 = ttir.empty() : tensor<1x640x1xf32>
    %6818 = "ttir.reshape"(%6816, %6817) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6819 = ttir.empty() : tensor<1x640x1xf32>
    %6820 = "ttir.add"(%6818, %46, %6819) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6821 = ttir.empty() : tensor<1x640x1xf32>
    %6822 = "ttir.rsqrt"(%6820, %6821) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6823 = ttir.empty() : tensor<1x640xf32>
    %6824 = "ttir.reshape"(%6822, %6823) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6825 = ttir.empty() : tensor<1x640x1xf32>
    %6826 = "ttir.reshape"(%6824, %6825) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6827 = ttir.empty() : tensor<1x640x3072xf32>
    %6828 = "ttir.broadcast"(%6826, %6827) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6829 = ttir.empty() : tensor<1x640x3072xf32>
    %6830 = "ttir.multiply"(%6810, %6828, %6829) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6831 = ttir.empty() : tensor<1x640x3072xbf16>
    %6832 = "ttir.typecast"(%6830, %6831) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6833 = ttir.empty() : tensor<1x640x3072xf32>
    %6834 = "ttir.typecast"(%6832, %6833) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6835 = ttir.empty() : tensor<1x640x3072xf32>
    %6836 = "ttir.multiply"(%6808, %6834, %6835) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6837 = ttir.empty() : tensor<1x640x3072xbf16>
    %6838 = "ttir.typecast"(%6836, %6837) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6839 = ttir.empty() : tensor<640x3072xbf16>
    %6840 = "ttir.reshape"(%6838, %6839) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %6841 = ttir.empty() : tensor<1x8192x3072xbf16>
    %6842 = "ttir.reshape"(%arg223, %6841) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %6843 = ttir.empty() : tensor<8192x3072xbf16>
    %6844 = "ttir.reshape"(%6842, %6843) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6845 = ttir.empty() : tensor<3072x8192xbf16>
    %6846 = "ttir.permute"(%6844, %6845) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6847 = "ttir.dot_general"(%6840, %6846) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %6848 = ttir.empty() : tensor<1x640x8192xbf16>
    %6849 = "ttir.reshape"(%6847, %6848) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6850 = ttir.empty() : tensor<1x640x8192xf32>
    %6851 = "ttir.typecast"(%6849, %6850) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6852 = ttir.empty() : tensor<1x640x8192xbf16>
    %6853 = "ttir.sigmoid"(%6849, %6852) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6854 = ttir.empty() : tensor<1x640x8192xf32>
    %6855 = "ttir.typecast"(%6853, %6854) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6856 = ttir.empty() : tensor<1x640x8192xf32>
    %6857 = "ttir.multiply"(%6851, %6855, %6856) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6858 = ttir.empty() : tensor<1x640x8192xbf16>
    %6859 = "ttir.typecast"(%6857, %6858) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6860 = ttir.empty() : tensor<1x640x8192xf32>
    %6861 = "ttir.typecast"(%6859, %6860) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6862 = ttir.empty() : tensor<1x8192x3072xbf16>
    %6863 = "ttir.reshape"(%arg218, %6862) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %6864 = ttir.empty() : tensor<8192x3072xbf16>
    %6865 = "ttir.reshape"(%6863, %6864) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6866 = ttir.empty() : tensor<3072x8192xbf16>
    %6867 = "ttir.permute"(%6865, %6866) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6868 = "ttir.dot_general"(%6840, %6867) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %6869 = ttir.empty() : tensor<1x640x8192xbf16>
    %6870 = "ttir.reshape"(%6868, %6869) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6871 = ttir.empty() : tensor<1x640x8192xf32>
    %6872 = "ttir.typecast"(%6870, %6871) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6873 = ttir.empty() : tensor<1x640x8192xf32>
    %6874 = "ttir.multiply"(%6861, %6872, %6873) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %6875 = ttir.empty() : tensor<1x640x8192xbf16>
    %6876 = "ttir.typecast"(%6874, %6875) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %6877 = ttir.empty() : tensor<640x8192xbf16>
    %6878 = "ttir.reshape"(%6876, %6877) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %6879 = ttir.empty() : tensor<1x3072x8192xbf16>
    %6880 = "ttir.reshape"(%arg217, %6879) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %6881 = ttir.empty() : tensor<3072x8192xbf16>
    %6882 = "ttir.reshape"(%6880, %6881) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %6883 = ttir.empty() : tensor<8192x3072xbf16>
    %6884 = "ttir.permute"(%6882, %6883) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %6885 = "ttir.dot_general"(%6878, %6884) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %6886 = ttir.empty() : tensor<1x640x3072xbf16>
    %6887 = "ttir.reshape"(%6885, %6886) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6888 = ttir.empty() : tensor<1x640x3072xbf16>
    %6889 = "ttir.add"(%6798, %6887, %6888) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6890 = ttir.empty() : tensor<1x640x3072xf32>
    %6891 = "ttir.typecast"(%6889, %6890) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6892 = ttir.empty() : tensor<1x640x3072xf32>
    %6893 = "ttir.pow"(%6891, %5, %6892) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6894 = ttir.empty() : tensor<1x640xf32>
    %6895 = "ttir.sum"(%6893, %6894) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6896 = ttir.empty() : tensor<1x640xf32>
    %6897 = "ttir.multiply"(%6895, %4, %6896) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6898 = ttir.empty() : tensor<1x640x1xf32>
    %6899 = "ttir.reshape"(%6897, %6898) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6900 = ttir.empty() : tensor<1x640x1xf32>
    %6901 = "ttir.add"(%6899, %46, %6900) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6902 = ttir.empty() : tensor<1x640x1xf32>
    %6903 = "ttir.rsqrt"(%6901, %6902) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6904 = ttir.empty() : tensor<1x640xf32>
    %6905 = "ttir.reshape"(%6903, %6904) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %6906 = ttir.empty() : tensor<1x640x1xf32>
    %6907 = "ttir.reshape"(%6905, %6906) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %6908 = ttir.empty() : tensor<1x640x3072xf32>
    %6909 = "ttir.broadcast"(%6907, %6908) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6910 = ttir.empty() : tensor<1x640x3072xf32>
    %6911 = "ttir.multiply"(%6891, %6909, %6910) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6912 = ttir.empty() : tensor<1x640x3072xbf16>
    %6913 = "ttir.typecast"(%6911, %6912) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6914 = ttir.empty() : tensor<1x640x3072xf32>
    %6915 = "ttir.typecast"(%6913, %6914) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6916 = ttir.empty() : tensor<1x640x3072xf32>
    %6917 = "ttir.multiply"(%6661, %6915, %6916) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6918 = ttir.empty() : tensor<1x640x3072xbf16>
    %6919 = "ttir.typecast"(%6917, %6918) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %6920 = ttir.empty() : tensor<640x3072xbf16>
    %6921 = "ttir.reshape"(%6919, %6920) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %6922 = ttir.empty() : tensor<1x1024x3072xbf16>
    %6923 = "ttir.reshape"(%arg216, %6922) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %6924 = ttir.empty() : tensor<1024x3072xbf16>
    %6925 = "ttir.reshape"(%6923, %6924) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %6926 = ttir.empty() : tensor<3072x1024xbf16>
    %6927 = "ttir.permute"(%6925, %6926) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %6928 = "ttir.dot_general"(%6921, %6927) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %6929 = ttir.empty() : tensor<1x640x8x128xbf16>
    %6930 = "ttir.reshape"(%6928, %6929) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %6931 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6932 = "ttir.permute"(%6930, %6931) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6933 = ttir.empty() : tensor<1x1x3072xbf16>
    %6934 = "ttir.reshape"(%arg233, %6933) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %6935 = ttir.empty() : tensor<3072xbf16>
    %6936 = "ttir.reshape"(%6934, %6935) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %6937 = ttir.empty() : tensor<3072xf32>
    %6938 = "ttir.typecast"(%6936, %6937) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %6939 = ttir.empty() : tensor<1x1x3072xf32>
    %6940 = "ttir.reshape"(%6938, %6939) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %6941 = ttir.empty() : tensor<1x640x3072xf32>
    %6942 = "ttir.broadcast"(%6940, %6941) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %6943 = ttir.empty() : tensor<1x3072x3072xbf16>
    %6944 = "ttir.reshape"(%arg230, %6943) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %6945 = ttir.empty() : tensor<3072x3072xbf16>
    %6946 = "ttir.reshape"(%6944, %6945) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6947 = ttir.empty() : tensor<3072x3072xbf16>
    %6948 = "ttir.permute"(%6946, %6947) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6949 = "ttir.dot_general"(%6921, %6948) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %6950 = ttir.empty() : tensor<1x640x24x128xbf16>
    %6951 = "ttir.reshape"(%6949, %6950) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %6952 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6953 = "ttir.permute"(%6951, %6952) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6954 = ttir.empty() : tensor<1x24x640x128xf32>
    %6955 = "ttir.typecast"(%6953, %6954) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6956 = ttir.empty() : tensor<1x24x640x128xf32>
    %6957 = "ttir.multiply"(%6955, %125, %6956) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6958 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6959 = "ttir.typecast"(%6957, %6958) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6960 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6961 = "ttir.slice_static"(%6953, %6960) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6962 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6963 = "ttir.neg"(%6961, %6962) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6964 = ttir.empty() : tensor<1x24x640x64xbf16>
    %6965 = "ttir.slice_static"(%6953, %6964) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %6966 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6967 = "ttir.concat"(%6963, %6965, %6966) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6968 = ttir.empty() : tensor<1x24x640x128xf32>
    %6969 = "ttir.typecast"(%6967, %6968) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6970 = ttir.empty() : tensor<1x24x640x128xf32>
    %6971 = "ttir.multiply"(%6969, %153, %6970) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %6972 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6973 = "ttir.typecast"(%6971, %6972) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6974 = ttir.empty() : tensor<1x24x640x128xbf16>
    %6975 = "ttir.add"(%6959, %6973, %6974) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %6976 = ttir.empty() : tensor<24x640x128xbf16>
    %6977 = "ttir.reshape"(%6975, %6976) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %6978 = ttir.empty() : tensor<1x1024x3072xbf16>
    %6979 = "ttir.reshape"(%arg229, %6978) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %6980 = ttir.empty() : tensor<1024x3072xbf16>
    %6981 = "ttir.reshape"(%6979, %6980) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %6982 = ttir.empty() : tensor<3072x1024xbf16>
    %6983 = "ttir.permute"(%6981, %6982) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %6984 = "ttir.dot_general"(%6921, %6983) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %6985 = ttir.empty() : tensor<1x640x8x128xbf16>
    %6986 = "ttir.reshape"(%6984, %6985) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %6987 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6988 = "ttir.permute"(%6986, %6987) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6989 = ttir.empty() : tensor<1x8x640x128xf32>
    %6990 = "ttir.typecast"(%6988, %6989) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6991 = ttir.empty() : tensor<1x8x640x128xf32>
    %6992 = "ttir.multiply"(%6990, %178, %6991) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %6993 = ttir.empty() : tensor<1x8x640x128xbf16>
    %6994 = "ttir.typecast"(%6992, %6993) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %6995 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6996 = "ttir.slice_static"(%6988, %6995) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6997 = ttir.empty() : tensor<1x8x640x64xbf16>
    %6998 = "ttir.neg"(%6996, %6997) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %6999 = ttir.empty() : tensor<1x8x640x64xbf16>
    %7000 = "ttir.slice_static"(%6988, %6999) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %7001 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7002 = "ttir.concat"(%6998, %7000, %7001) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7003 = ttir.empty() : tensor<1x8x640x128xf32>
    %7004 = "ttir.typecast"(%7002, %7003) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7005 = ttir.empty() : tensor<1x8x640x128xf32>
    %7006 = "ttir.multiply"(%7004, %196, %7005) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7007 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7008 = "ttir.typecast"(%7006, %7007) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7009 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7010 = "ttir.add"(%6994, %7008, %7009) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7011 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %7012 = "ttir.reshape"(%7010, %7011) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %7013 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %7014 = "ttir.broadcast"(%7012, %7013) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %7015 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7016 = "ttir.reshape"(%7014, %7015) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7017 = ttir.empty() : tensor<1x24x128x640xbf16>
    %7018 = "ttir.permute"(%7016, %7017) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %7019 = ttir.empty() : tensor<24x128x640xbf16>
    %7020 = "ttir.reshape"(%7018, %7019) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %7021 = "ttir.dot_general"(%6977, %7020) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %7022 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7023 = "ttir.reshape"(%7021, %7022) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7024 = ttir.empty() : tensor<1x24x640x640xf32>
    %7025 = "ttir.typecast"(%7023, %7024) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7026 = ttir.empty() : tensor<1x24x640x640xf32>
    %7027 = "ttir.multiply"(%7025, %221, %7026) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7028 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7029 = "ttir.typecast"(%7027, %7028) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7030 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7031 = "ttir.add"(%7029, %285, %7030) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7032 = ttir.empty() : tensor<1x24x640x640xf32>
    %7033 = "ttir.typecast"(%7031, %7032) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7034 = ttir.empty() : tensor<1x24x640xf32>
    %7035 = "ttir.max"(%7033, %7034) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %7036 = ttir.empty() : tensor<1x24x640x1xf32>
    %7037 = "ttir.reshape"(%7035, %7036) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %7038 = ttir.empty() : tensor<1x24x640x640xf32>
    %7039 = "ttir.broadcast"(%7037, %7038) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7040 = ttir.empty() : tensor<1x24x640x640xf32>
    %7041 = "ttir.subtract"(%7033, %7039, %7040) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7042 = ttir.empty() : tensor<1x24x640x640xf32>
    %7043 = "ttir.exp"(%7041, %7042) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7044 = ttir.empty() : tensor<1x24x640xf32>
    %7045 = "ttir.sum"(%7043, %7044) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %7046 = ttir.empty() : tensor<1x24x640x1xf32>
    %7047 = "ttir.reshape"(%7045, %7046) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %7048 = ttir.empty() : tensor<1x24x640x640xf32>
    %7049 = "ttir.broadcast"(%7047, %7048) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7050 = ttir.empty() : tensor<1x24x640x640xf32>
    %7051 = "ttir.div"(%7043, %7049, %7050) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7052 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7053 = "ttir.typecast"(%7051, %7052) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7054 = ttir.empty() : tensor<24x640x640xbf16>
    %7055 = "ttir.reshape"(%7053, %7054) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %7056 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %7057 = "ttir.reshape"(%6932, %7056) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %7058 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %7059 = "ttir.broadcast"(%7057, %7058) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %7060 = ttir.empty() : tensor<24x640x128xbf16>
    %7061 = "ttir.reshape"(%7059, %7060) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7062 = "ttir.dot_general"(%7055, %7061) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7063 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7064 = "ttir.reshape"(%7062, %7063) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7065 = ttir.empty() : tensor<1x640x24x128xbf16>
    %7066 = "ttir.permute"(%7064, %7065) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %7067 = ttir.empty() : tensor<640x3072xbf16>
    %7068 = "ttir.reshape"(%7066, %7067) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7069 = ttir.empty() : tensor<1x3072x3072xbf16>
    %7070 = "ttir.reshape"(%arg228, %7069) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %7071 = ttir.empty() : tensor<3072x3072xbf16>
    %7072 = "ttir.reshape"(%7070, %7071) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7073 = ttir.empty() : tensor<3072x3072xbf16>
    %7074 = "ttir.permute"(%7072, %7073) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7075 = "ttir.dot_general"(%7068, %7074) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %7076 = ttir.empty() : tensor<1x640x3072xbf16>
    %7077 = "ttir.reshape"(%7075, %7076) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7078 = ttir.empty() : tensor<1x640x3072xbf16>
    %7079 = "ttir.add"(%6889, %7077, %7078) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7080 = ttir.empty() : tensor<1x1x3072xbf16>
    %7081 = "ttir.reshape"(%arg231, %7080) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %7082 = ttir.empty() : tensor<3072xbf16>
    %7083 = "ttir.reshape"(%7081, %7082) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %7084 = ttir.empty() : tensor<3072xf32>
    %7085 = "ttir.typecast"(%7083, %7084) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %7086 = ttir.empty() : tensor<1x1x3072xf32>
    %7087 = "ttir.reshape"(%7085, %7086) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %7088 = ttir.empty() : tensor<1x640x3072xf32>
    %7089 = "ttir.broadcast"(%7087, %7088) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7090 = ttir.empty() : tensor<1x640x3072xf32>
    %7091 = "ttir.typecast"(%7079, %7090) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7092 = ttir.empty() : tensor<1x640x3072xf32>
    %7093 = "ttir.pow"(%7091, %5, %7092) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7094 = ttir.empty() : tensor<1x640xf32>
    %7095 = "ttir.sum"(%7093, %7094) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7096 = ttir.empty() : tensor<1x640xf32>
    %7097 = "ttir.multiply"(%7095, %4, %7096) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7098 = ttir.empty() : tensor<1x640x1xf32>
    %7099 = "ttir.reshape"(%7097, %7098) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7100 = ttir.empty() : tensor<1x640x1xf32>
    %7101 = "ttir.add"(%7099, %46, %7100) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7102 = ttir.empty() : tensor<1x640x1xf32>
    %7103 = "ttir.rsqrt"(%7101, %7102) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7104 = ttir.empty() : tensor<1x640xf32>
    %7105 = "ttir.reshape"(%7103, %7104) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7106 = ttir.empty() : tensor<1x640x1xf32>
    %7107 = "ttir.reshape"(%7105, %7106) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7108 = ttir.empty() : tensor<1x640x3072xf32>
    %7109 = "ttir.broadcast"(%7107, %7108) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7110 = ttir.empty() : tensor<1x640x3072xf32>
    %7111 = "ttir.multiply"(%7091, %7109, %7110) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7112 = ttir.empty() : tensor<1x640x3072xbf16>
    %7113 = "ttir.typecast"(%7111, %7112) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7114 = ttir.empty() : tensor<1x640x3072xf32>
    %7115 = "ttir.typecast"(%7113, %7114) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7116 = ttir.empty() : tensor<1x640x3072xf32>
    %7117 = "ttir.multiply"(%7089, %7115, %7116) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7118 = ttir.empty() : tensor<1x640x3072xbf16>
    %7119 = "ttir.typecast"(%7117, %7118) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7120 = ttir.empty() : tensor<640x3072xbf16>
    %7121 = "ttir.reshape"(%7119, %7120) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7122 = ttir.empty() : tensor<1x8192x3072xbf16>
    %7123 = "ttir.reshape"(%arg232, %7122) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %7124 = ttir.empty() : tensor<8192x3072xbf16>
    %7125 = "ttir.reshape"(%7123, %7124) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7126 = ttir.empty() : tensor<3072x8192xbf16>
    %7127 = "ttir.permute"(%7125, %7126) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7128 = "ttir.dot_general"(%7121, %7127) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %7129 = ttir.empty() : tensor<1x640x8192xbf16>
    %7130 = "ttir.reshape"(%7128, %7129) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7131 = ttir.empty() : tensor<1x640x8192xf32>
    %7132 = "ttir.typecast"(%7130, %7131) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7133 = ttir.empty() : tensor<1x640x8192xbf16>
    %7134 = "ttir.sigmoid"(%7130, %7133) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7135 = ttir.empty() : tensor<1x640x8192xf32>
    %7136 = "ttir.typecast"(%7134, %7135) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7137 = ttir.empty() : tensor<1x640x8192xf32>
    %7138 = "ttir.multiply"(%7132, %7136, %7137) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7139 = ttir.empty() : tensor<1x640x8192xbf16>
    %7140 = "ttir.typecast"(%7138, %7139) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7141 = ttir.empty() : tensor<1x640x8192xf32>
    %7142 = "ttir.typecast"(%7140, %7141) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7143 = ttir.empty() : tensor<1x8192x3072xbf16>
    %7144 = "ttir.reshape"(%arg227, %7143) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %7145 = ttir.empty() : tensor<8192x3072xbf16>
    %7146 = "ttir.reshape"(%7144, %7145) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7147 = ttir.empty() : tensor<3072x8192xbf16>
    %7148 = "ttir.permute"(%7146, %7147) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7149 = "ttir.dot_general"(%7121, %7148) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %7150 = ttir.empty() : tensor<1x640x8192xbf16>
    %7151 = "ttir.reshape"(%7149, %7150) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7152 = ttir.empty() : tensor<1x640x8192xf32>
    %7153 = "ttir.typecast"(%7151, %7152) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7154 = ttir.empty() : tensor<1x640x8192xf32>
    %7155 = "ttir.multiply"(%7142, %7153, %7154) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7156 = ttir.empty() : tensor<1x640x8192xbf16>
    %7157 = "ttir.typecast"(%7155, %7156) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7158 = ttir.empty() : tensor<640x8192xbf16>
    %7159 = "ttir.reshape"(%7157, %7158) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %7160 = ttir.empty() : tensor<1x3072x8192xbf16>
    %7161 = "ttir.reshape"(%arg226, %7160) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %7162 = ttir.empty() : tensor<3072x8192xbf16>
    %7163 = "ttir.reshape"(%7161, %7162) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7164 = ttir.empty() : tensor<8192x3072xbf16>
    %7165 = "ttir.permute"(%7163, %7164) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7166 = "ttir.dot_general"(%7159, %7165) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %7167 = ttir.empty() : tensor<1x640x3072xbf16>
    %7168 = "ttir.reshape"(%7166, %7167) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7169 = ttir.empty() : tensor<1x640x3072xbf16>
    %7170 = "ttir.add"(%7079, %7168, %7169) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7171 = ttir.empty() : tensor<1x640x3072xf32>
    %7172 = "ttir.typecast"(%7170, %7171) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7173 = ttir.empty() : tensor<1x640x3072xf32>
    %7174 = "ttir.pow"(%7172, %5, %7173) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7175 = ttir.empty() : tensor<1x640xf32>
    %7176 = "ttir.sum"(%7174, %7175) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7177 = ttir.empty() : tensor<1x640xf32>
    %7178 = "ttir.multiply"(%7176, %4, %7177) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7179 = ttir.empty() : tensor<1x640x1xf32>
    %7180 = "ttir.reshape"(%7178, %7179) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7181 = ttir.empty() : tensor<1x640x1xf32>
    %7182 = "ttir.add"(%7180, %46, %7181) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7183 = ttir.empty() : tensor<1x640x1xf32>
    %7184 = "ttir.rsqrt"(%7182, %7183) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7185 = ttir.empty() : tensor<1x640xf32>
    %7186 = "ttir.reshape"(%7184, %7185) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7187 = ttir.empty() : tensor<1x640x1xf32>
    %7188 = "ttir.reshape"(%7186, %7187) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7189 = ttir.empty() : tensor<1x640x3072xf32>
    %7190 = "ttir.broadcast"(%7188, %7189) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7191 = ttir.empty() : tensor<1x640x3072xf32>
    %7192 = "ttir.multiply"(%7172, %7190, %7191) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7193 = ttir.empty() : tensor<1x640x3072xbf16>
    %7194 = "ttir.typecast"(%7192, %7193) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7195 = ttir.empty() : tensor<1x640x3072xf32>
    %7196 = "ttir.typecast"(%7194, %7195) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7197 = ttir.empty() : tensor<1x640x3072xf32>
    %7198 = "ttir.multiply"(%6942, %7196, %7197) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7199 = ttir.empty() : tensor<1x640x3072xbf16>
    %7200 = "ttir.typecast"(%7198, %7199) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7201 = ttir.empty() : tensor<640x3072xbf16>
    %7202 = "ttir.reshape"(%7200, %7201) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7203 = ttir.empty() : tensor<1x1024x3072xbf16>
    %7204 = "ttir.reshape"(%arg225, %7203) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %7205 = ttir.empty() : tensor<1024x3072xbf16>
    %7206 = "ttir.reshape"(%7204, %7205) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %7207 = ttir.empty() : tensor<3072x1024xbf16>
    %7208 = "ttir.permute"(%7206, %7207) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %7209 = "ttir.dot_general"(%7202, %7208) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %7210 = ttir.empty() : tensor<1x640x8x128xbf16>
    %7211 = "ttir.reshape"(%7209, %7210) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %7212 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7213 = "ttir.permute"(%7211, %7212) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7214 = ttir.empty() : tensor<1x1x3072xbf16>
    %7215 = "ttir.reshape"(%arg242, %7214) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %7216 = ttir.empty() : tensor<3072xbf16>
    %7217 = "ttir.reshape"(%7215, %7216) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %7218 = ttir.empty() : tensor<3072xf32>
    %7219 = "ttir.typecast"(%7217, %7218) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %7220 = ttir.empty() : tensor<1x1x3072xf32>
    %7221 = "ttir.reshape"(%7219, %7220) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %7222 = ttir.empty() : tensor<1x640x3072xf32>
    %7223 = "ttir.broadcast"(%7221, %7222) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7224 = ttir.empty() : tensor<1x3072x3072xbf16>
    %7225 = "ttir.reshape"(%arg239, %7224) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %7226 = ttir.empty() : tensor<3072x3072xbf16>
    %7227 = "ttir.reshape"(%7225, %7226) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7228 = ttir.empty() : tensor<3072x3072xbf16>
    %7229 = "ttir.permute"(%7227, %7228) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7230 = "ttir.dot_general"(%7202, %7229) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %7231 = ttir.empty() : tensor<1x640x24x128xbf16>
    %7232 = "ttir.reshape"(%7230, %7231) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %7233 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7234 = "ttir.permute"(%7232, %7233) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7235 = ttir.empty() : tensor<1x24x640x128xf32>
    %7236 = "ttir.typecast"(%7234, %7235) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7237 = ttir.empty() : tensor<1x24x640x128xf32>
    %7238 = "ttir.multiply"(%7236, %125, %7237) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7239 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7240 = "ttir.typecast"(%7238, %7239) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7241 = ttir.empty() : tensor<1x24x640x64xbf16>
    %7242 = "ttir.slice_static"(%7234, %7241) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %7243 = ttir.empty() : tensor<1x24x640x64xbf16>
    %7244 = "ttir.neg"(%7242, %7243) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %7245 = ttir.empty() : tensor<1x24x640x64xbf16>
    %7246 = "ttir.slice_static"(%7234, %7245) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %7247 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7248 = "ttir.concat"(%7244, %7246, %7247) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7249 = ttir.empty() : tensor<1x24x640x128xf32>
    %7250 = "ttir.typecast"(%7248, %7249) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7251 = ttir.empty() : tensor<1x24x640x128xf32>
    %7252 = "ttir.multiply"(%7250, %153, %7251) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7253 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7254 = "ttir.typecast"(%7252, %7253) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7255 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7256 = "ttir.add"(%7240, %7254, %7255) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7257 = ttir.empty() : tensor<24x640x128xbf16>
    %7258 = "ttir.reshape"(%7256, %7257) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7259 = ttir.empty() : tensor<1x1024x3072xbf16>
    %7260 = "ttir.reshape"(%arg238, %7259) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %7261 = ttir.empty() : tensor<1024x3072xbf16>
    %7262 = "ttir.reshape"(%7260, %7261) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %7263 = ttir.empty() : tensor<3072x1024xbf16>
    %7264 = "ttir.permute"(%7262, %7263) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %7265 = "ttir.dot_general"(%7202, %7264) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %7266 = ttir.empty() : tensor<1x640x8x128xbf16>
    %7267 = "ttir.reshape"(%7265, %7266) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %7268 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7269 = "ttir.permute"(%7267, %7268) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7270 = ttir.empty() : tensor<1x8x640x128xf32>
    %7271 = "ttir.typecast"(%7269, %7270) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7272 = ttir.empty() : tensor<1x8x640x128xf32>
    %7273 = "ttir.multiply"(%7271, %178, %7272) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7274 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7275 = "ttir.typecast"(%7273, %7274) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7276 = ttir.empty() : tensor<1x8x640x64xbf16>
    %7277 = "ttir.slice_static"(%7269, %7276) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %7278 = ttir.empty() : tensor<1x8x640x64xbf16>
    %7279 = "ttir.neg"(%7277, %7278) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %7280 = ttir.empty() : tensor<1x8x640x64xbf16>
    %7281 = "ttir.slice_static"(%7269, %7280) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %7282 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7283 = "ttir.concat"(%7279, %7281, %7282) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7284 = ttir.empty() : tensor<1x8x640x128xf32>
    %7285 = "ttir.typecast"(%7283, %7284) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7286 = ttir.empty() : tensor<1x8x640x128xf32>
    %7287 = "ttir.multiply"(%7285, %196, %7286) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7288 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7289 = "ttir.typecast"(%7287, %7288) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7290 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7291 = "ttir.add"(%7275, %7289, %7290) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7292 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %7293 = "ttir.reshape"(%7291, %7292) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %7294 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %7295 = "ttir.broadcast"(%7293, %7294) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %7296 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7297 = "ttir.reshape"(%7295, %7296) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7298 = ttir.empty() : tensor<1x24x128x640xbf16>
    %7299 = "ttir.permute"(%7297, %7298) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %7300 = ttir.empty() : tensor<24x128x640xbf16>
    %7301 = "ttir.reshape"(%7299, %7300) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %7302 = "ttir.dot_general"(%7258, %7301) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %7303 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7304 = "ttir.reshape"(%7302, %7303) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7305 = ttir.empty() : tensor<1x24x640x640xf32>
    %7306 = "ttir.typecast"(%7304, %7305) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7307 = ttir.empty() : tensor<1x24x640x640xf32>
    %7308 = "ttir.multiply"(%7306, %221, %7307) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7309 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7310 = "ttir.typecast"(%7308, %7309) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7311 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7312 = "ttir.add"(%7310, %285, %7311) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7313 = ttir.empty() : tensor<1x24x640x640xf32>
    %7314 = "ttir.typecast"(%7312, %7313) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7315 = ttir.empty() : tensor<1x24x640xf32>
    %7316 = "ttir.max"(%7314, %7315) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %7317 = ttir.empty() : tensor<1x24x640x1xf32>
    %7318 = "ttir.reshape"(%7316, %7317) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %7319 = ttir.empty() : tensor<1x24x640x640xf32>
    %7320 = "ttir.broadcast"(%7318, %7319) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7321 = ttir.empty() : tensor<1x24x640x640xf32>
    %7322 = "ttir.subtract"(%7314, %7320, %7321) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7323 = ttir.empty() : tensor<1x24x640x640xf32>
    %7324 = "ttir.exp"(%7322, %7323) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7325 = ttir.empty() : tensor<1x24x640xf32>
    %7326 = "ttir.sum"(%7324, %7325) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %7327 = ttir.empty() : tensor<1x24x640x1xf32>
    %7328 = "ttir.reshape"(%7326, %7327) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %7329 = ttir.empty() : tensor<1x24x640x640xf32>
    %7330 = "ttir.broadcast"(%7328, %7329) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7331 = ttir.empty() : tensor<1x24x640x640xf32>
    %7332 = "ttir.div"(%7324, %7330, %7331) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7333 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7334 = "ttir.typecast"(%7332, %7333) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7335 = ttir.empty() : tensor<24x640x640xbf16>
    %7336 = "ttir.reshape"(%7334, %7335) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %7337 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %7338 = "ttir.reshape"(%7213, %7337) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %7339 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %7340 = "ttir.broadcast"(%7338, %7339) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %7341 = ttir.empty() : tensor<24x640x128xbf16>
    %7342 = "ttir.reshape"(%7340, %7341) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7343 = "ttir.dot_general"(%7336, %7342) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7344 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7345 = "ttir.reshape"(%7343, %7344) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7346 = ttir.empty() : tensor<1x640x24x128xbf16>
    %7347 = "ttir.permute"(%7345, %7346) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %7348 = ttir.empty() : tensor<640x3072xbf16>
    %7349 = "ttir.reshape"(%7347, %7348) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7350 = ttir.empty() : tensor<1x3072x3072xbf16>
    %7351 = "ttir.reshape"(%arg237, %7350) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %7352 = ttir.empty() : tensor<3072x3072xbf16>
    %7353 = "ttir.reshape"(%7351, %7352) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7354 = ttir.empty() : tensor<3072x3072xbf16>
    %7355 = "ttir.permute"(%7353, %7354) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7356 = "ttir.dot_general"(%7349, %7355) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %7357 = ttir.empty() : tensor<1x640x3072xbf16>
    %7358 = "ttir.reshape"(%7356, %7357) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7359 = ttir.empty() : tensor<1x640x3072xbf16>
    %7360 = "ttir.add"(%7170, %7358, %7359) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7361 = ttir.empty() : tensor<1x1x3072xbf16>
    %7362 = "ttir.reshape"(%arg240, %7361) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %7363 = ttir.empty() : tensor<3072xbf16>
    %7364 = "ttir.reshape"(%7362, %7363) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %7365 = ttir.empty() : tensor<3072xf32>
    %7366 = "ttir.typecast"(%7364, %7365) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %7367 = ttir.empty() : tensor<1x1x3072xf32>
    %7368 = "ttir.reshape"(%7366, %7367) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %7369 = ttir.empty() : tensor<1x640x3072xf32>
    %7370 = "ttir.broadcast"(%7368, %7369) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7371 = ttir.empty() : tensor<1x640x3072xf32>
    %7372 = "ttir.typecast"(%7360, %7371) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7373 = ttir.empty() : tensor<1x640x3072xf32>
    %7374 = "ttir.pow"(%7372, %5, %7373) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7375 = ttir.empty() : tensor<1x640xf32>
    %7376 = "ttir.sum"(%7374, %7375) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7377 = ttir.empty() : tensor<1x640xf32>
    %7378 = "ttir.multiply"(%7376, %4, %7377) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7379 = ttir.empty() : tensor<1x640x1xf32>
    %7380 = "ttir.reshape"(%7378, %7379) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7381 = ttir.empty() : tensor<1x640x1xf32>
    %7382 = "ttir.add"(%7380, %46, %7381) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7383 = ttir.empty() : tensor<1x640x1xf32>
    %7384 = "ttir.rsqrt"(%7382, %7383) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7385 = ttir.empty() : tensor<1x640xf32>
    %7386 = "ttir.reshape"(%7384, %7385) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7387 = ttir.empty() : tensor<1x640x1xf32>
    %7388 = "ttir.reshape"(%7386, %7387) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7389 = ttir.empty() : tensor<1x640x3072xf32>
    %7390 = "ttir.broadcast"(%7388, %7389) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7391 = ttir.empty() : tensor<1x640x3072xf32>
    %7392 = "ttir.multiply"(%7372, %7390, %7391) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7393 = ttir.empty() : tensor<1x640x3072xbf16>
    %7394 = "ttir.typecast"(%7392, %7393) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7395 = ttir.empty() : tensor<1x640x3072xf32>
    %7396 = "ttir.typecast"(%7394, %7395) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7397 = ttir.empty() : tensor<1x640x3072xf32>
    %7398 = "ttir.multiply"(%7370, %7396, %7397) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7399 = ttir.empty() : tensor<1x640x3072xbf16>
    %7400 = "ttir.typecast"(%7398, %7399) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7401 = ttir.empty() : tensor<640x3072xbf16>
    %7402 = "ttir.reshape"(%7400, %7401) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7403 = ttir.empty() : tensor<1x8192x3072xbf16>
    %7404 = "ttir.reshape"(%arg241, %7403) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %7405 = ttir.empty() : tensor<8192x3072xbf16>
    %7406 = "ttir.reshape"(%7404, %7405) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7407 = ttir.empty() : tensor<3072x8192xbf16>
    %7408 = "ttir.permute"(%7406, %7407) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7409 = "ttir.dot_general"(%7402, %7408) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %7410 = ttir.empty() : tensor<1x640x8192xbf16>
    %7411 = "ttir.reshape"(%7409, %7410) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7412 = ttir.empty() : tensor<1x640x8192xf32>
    %7413 = "ttir.typecast"(%7411, %7412) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7414 = ttir.empty() : tensor<1x640x8192xbf16>
    %7415 = "ttir.sigmoid"(%7411, %7414) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7416 = ttir.empty() : tensor<1x640x8192xf32>
    %7417 = "ttir.typecast"(%7415, %7416) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7418 = ttir.empty() : tensor<1x640x8192xf32>
    %7419 = "ttir.multiply"(%7413, %7417, %7418) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7420 = ttir.empty() : tensor<1x640x8192xbf16>
    %7421 = "ttir.typecast"(%7419, %7420) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7422 = ttir.empty() : tensor<1x640x8192xf32>
    %7423 = "ttir.typecast"(%7421, %7422) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7424 = ttir.empty() : tensor<1x8192x3072xbf16>
    %7425 = "ttir.reshape"(%arg236, %7424) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %7426 = ttir.empty() : tensor<8192x3072xbf16>
    %7427 = "ttir.reshape"(%7425, %7426) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7428 = ttir.empty() : tensor<3072x8192xbf16>
    %7429 = "ttir.permute"(%7427, %7428) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7430 = "ttir.dot_general"(%7402, %7429) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %7431 = ttir.empty() : tensor<1x640x8192xbf16>
    %7432 = "ttir.reshape"(%7430, %7431) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7433 = ttir.empty() : tensor<1x640x8192xf32>
    %7434 = "ttir.typecast"(%7432, %7433) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7435 = ttir.empty() : tensor<1x640x8192xf32>
    %7436 = "ttir.multiply"(%7423, %7434, %7435) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7437 = ttir.empty() : tensor<1x640x8192xbf16>
    %7438 = "ttir.typecast"(%7436, %7437) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7439 = ttir.empty() : tensor<640x8192xbf16>
    %7440 = "ttir.reshape"(%7438, %7439) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %7441 = ttir.empty() : tensor<1x3072x8192xbf16>
    %7442 = "ttir.reshape"(%arg235, %7441) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %7443 = ttir.empty() : tensor<3072x8192xbf16>
    %7444 = "ttir.reshape"(%7442, %7443) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7445 = ttir.empty() : tensor<8192x3072xbf16>
    %7446 = "ttir.permute"(%7444, %7445) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7447 = "ttir.dot_general"(%7440, %7446) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %7448 = ttir.empty() : tensor<1x640x3072xbf16>
    %7449 = "ttir.reshape"(%7447, %7448) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7450 = ttir.empty() : tensor<1x640x3072xbf16>
    %7451 = "ttir.add"(%7360, %7449, %7450) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7452 = ttir.empty() : tensor<1x640x3072xf32>
    %7453 = "ttir.typecast"(%7451, %7452) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7454 = ttir.empty() : tensor<1x640x3072xf32>
    %7455 = "ttir.pow"(%7453, %5, %7454) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7456 = ttir.empty() : tensor<1x640xf32>
    %7457 = "ttir.sum"(%7455, %7456) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7458 = ttir.empty() : tensor<1x640xf32>
    %7459 = "ttir.multiply"(%7457, %4, %7458) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7460 = ttir.empty() : tensor<1x640x1xf32>
    %7461 = "ttir.reshape"(%7459, %7460) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7462 = ttir.empty() : tensor<1x640x1xf32>
    %7463 = "ttir.add"(%7461, %46, %7462) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7464 = ttir.empty() : tensor<1x640x1xf32>
    %7465 = "ttir.rsqrt"(%7463, %7464) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7466 = ttir.empty() : tensor<1x640xf32>
    %7467 = "ttir.reshape"(%7465, %7466) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7468 = ttir.empty() : tensor<1x640x1xf32>
    %7469 = "ttir.reshape"(%7467, %7468) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7470 = ttir.empty() : tensor<1x640x3072xf32>
    %7471 = "ttir.broadcast"(%7469, %7470) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7472 = ttir.empty() : tensor<1x640x3072xf32>
    %7473 = "ttir.multiply"(%7453, %7471, %7472) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7474 = ttir.empty() : tensor<1x640x3072xbf16>
    %7475 = "ttir.typecast"(%7473, %7474) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7476 = ttir.empty() : tensor<1x640x3072xf32>
    %7477 = "ttir.typecast"(%7475, %7476) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7478 = ttir.empty() : tensor<1x640x3072xf32>
    %7479 = "ttir.multiply"(%7223, %7477, %7478) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7480 = ttir.empty() : tensor<1x640x3072xbf16>
    %7481 = "ttir.typecast"(%7479, %7480) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7482 = ttir.empty() : tensor<640x3072xbf16>
    %7483 = "ttir.reshape"(%7481, %7482) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7484 = ttir.empty() : tensor<1x1024x3072xbf16>
    %7485 = "ttir.reshape"(%arg234, %7484) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %7486 = ttir.empty() : tensor<1024x3072xbf16>
    %7487 = "ttir.reshape"(%7485, %7486) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %7488 = ttir.empty() : tensor<3072x1024xbf16>
    %7489 = "ttir.permute"(%7487, %7488) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %7490 = "ttir.dot_general"(%7483, %7489) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %7491 = ttir.empty() : tensor<1x640x8x128xbf16>
    %7492 = "ttir.reshape"(%7490, %7491) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %7493 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7494 = "ttir.permute"(%7492, %7493) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7495 = ttir.empty() : tensor<1x1x3072xbf16>
    %7496 = "ttir.reshape"(%arg251, %7495) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %7497 = ttir.empty() : tensor<3072xbf16>
    %7498 = "ttir.reshape"(%7496, %7497) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %7499 = ttir.empty() : tensor<3072xf32>
    %7500 = "ttir.typecast"(%7498, %7499) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %7501 = ttir.empty() : tensor<1x1x3072xf32>
    %7502 = "ttir.reshape"(%7500, %7501) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %7503 = ttir.empty() : tensor<1x640x3072xf32>
    %7504 = "ttir.broadcast"(%7502, %7503) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7505 = ttir.empty() : tensor<1x3072x3072xbf16>
    %7506 = "ttir.reshape"(%arg248, %7505) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %7507 = ttir.empty() : tensor<3072x3072xbf16>
    %7508 = "ttir.reshape"(%7506, %7507) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7509 = ttir.empty() : tensor<3072x3072xbf16>
    %7510 = "ttir.permute"(%7508, %7509) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7511 = "ttir.dot_general"(%7483, %7510) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %7512 = ttir.empty() : tensor<1x640x24x128xbf16>
    %7513 = "ttir.reshape"(%7511, %7512) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %7514 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7515 = "ttir.permute"(%7513, %7514) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7516 = ttir.empty() : tensor<1x24x640x128xf32>
    %7517 = "ttir.typecast"(%7515, %7516) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7518 = ttir.empty() : tensor<1x24x640x128xf32>
    %7519 = "ttir.multiply"(%7517, %125, %7518) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7520 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7521 = "ttir.typecast"(%7519, %7520) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7522 = ttir.empty() : tensor<1x24x640x64xbf16>
    %7523 = "ttir.slice_static"(%7515, %7522) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %7524 = ttir.empty() : tensor<1x24x640x64xbf16>
    %7525 = "ttir.neg"(%7523, %7524) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %7526 = ttir.empty() : tensor<1x24x640x64xbf16>
    %7527 = "ttir.slice_static"(%7515, %7526) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %7528 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7529 = "ttir.concat"(%7525, %7527, %7528) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7530 = ttir.empty() : tensor<1x24x640x128xf32>
    %7531 = "ttir.typecast"(%7529, %7530) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7532 = ttir.empty() : tensor<1x24x640x128xf32>
    %7533 = "ttir.multiply"(%7531, %153, %7532) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7534 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7535 = "ttir.typecast"(%7533, %7534) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7536 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7537 = "ttir.add"(%7521, %7535, %7536) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7538 = ttir.empty() : tensor<24x640x128xbf16>
    %7539 = "ttir.reshape"(%7537, %7538) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7540 = ttir.empty() : tensor<1x1024x3072xbf16>
    %7541 = "ttir.reshape"(%arg247, %7540) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %7542 = ttir.empty() : tensor<1024x3072xbf16>
    %7543 = "ttir.reshape"(%7541, %7542) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %7544 = ttir.empty() : tensor<3072x1024xbf16>
    %7545 = "ttir.permute"(%7543, %7544) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %7546 = "ttir.dot_general"(%7483, %7545) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %7547 = ttir.empty() : tensor<1x640x8x128xbf16>
    %7548 = "ttir.reshape"(%7546, %7547) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %7549 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7550 = "ttir.permute"(%7548, %7549) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7551 = ttir.empty() : tensor<1x8x640x128xf32>
    %7552 = "ttir.typecast"(%7550, %7551) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7553 = ttir.empty() : tensor<1x8x640x128xf32>
    %7554 = "ttir.multiply"(%7552, %178, %7553) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7555 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7556 = "ttir.typecast"(%7554, %7555) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7557 = ttir.empty() : tensor<1x8x640x64xbf16>
    %7558 = "ttir.slice_static"(%7550, %7557) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %7559 = ttir.empty() : tensor<1x8x640x64xbf16>
    %7560 = "ttir.neg"(%7558, %7559) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %7561 = ttir.empty() : tensor<1x8x640x64xbf16>
    %7562 = "ttir.slice_static"(%7550, %7561) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %7563 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7564 = "ttir.concat"(%7560, %7562, %7563) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7565 = ttir.empty() : tensor<1x8x640x128xf32>
    %7566 = "ttir.typecast"(%7564, %7565) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7567 = ttir.empty() : tensor<1x8x640x128xf32>
    %7568 = "ttir.multiply"(%7566, %196, %7567) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7569 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7570 = "ttir.typecast"(%7568, %7569) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7571 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7572 = "ttir.add"(%7556, %7570, %7571) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7573 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %7574 = "ttir.reshape"(%7572, %7573) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %7575 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %7576 = "ttir.broadcast"(%7574, %7575) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %7577 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7578 = "ttir.reshape"(%7576, %7577) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7579 = ttir.empty() : tensor<1x24x128x640xbf16>
    %7580 = "ttir.permute"(%7578, %7579) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %7581 = ttir.empty() : tensor<24x128x640xbf16>
    %7582 = "ttir.reshape"(%7580, %7581) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %7583 = "ttir.dot_general"(%7539, %7582) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %7584 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7585 = "ttir.reshape"(%7583, %7584) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7586 = ttir.empty() : tensor<1x24x640x640xf32>
    %7587 = "ttir.typecast"(%7585, %7586) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7588 = ttir.empty() : tensor<1x24x640x640xf32>
    %7589 = "ttir.multiply"(%7587, %221, %7588) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7590 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7591 = "ttir.typecast"(%7589, %7590) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7592 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7593 = "ttir.add"(%7591, %285, %7592) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7594 = ttir.empty() : tensor<1x24x640x640xf32>
    %7595 = "ttir.typecast"(%7593, %7594) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7596 = ttir.empty() : tensor<1x24x640xf32>
    %7597 = "ttir.max"(%7595, %7596) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %7598 = ttir.empty() : tensor<1x24x640x1xf32>
    %7599 = "ttir.reshape"(%7597, %7598) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %7600 = ttir.empty() : tensor<1x24x640x640xf32>
    %7601 = "ttir.broadcast"(%7599, %7600) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7602 = ttir.empty() : tensor<1x24x640x640xf32>
    %7603 = "ttir.subtract"(%7595, %7601, %7602) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7604 = ttir.empty() : tensor<1x24x640x640xf32>
    %7605 = "ttir.exp"(%7603, %7604) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7606 = ttir.empty() : tensor<1x24x640xf32>
    %7607 = "ttir.sum"(%7605, %7606) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %7608 = ttir.empty() : tensor<1x24x640x1xf32>
    %7609 = "ttir.reshape"(%7607, %7608) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %7610 = ttir.empty() : tensor<1x24x640x640xf32>
    %7611 = "ttir.broadcast"(%7609, %7610) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7612 = ttir.empty() : tensor<1x24x640x640xf32>
    %7613 = "ttir.div"(%7605, %7611, %7612) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7614 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7615 = "ttir.typecast"(%7613, %7614) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7616 = ttir.empty() : tensor<24x640x640xbf16>
    %7617 = "ttir.reshape"(%7615, %7616) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %7618 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %7619 = "ttir.reshape"(%7494, %7618) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %7620 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %7621 = "ttir.broadcast"(%7619, %7620) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %7622 = ttir.empty() : tensor<24x640x128xbf16>
    %7623 = "ttir.reshape"(%7621, %7622) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7624 = "ttir.dot_general"(%7617, %7623) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7625 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7626 = "ttir.reshape"(%7624, %7625) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7627 = ttir.empty() : tensor<1x640x24x128xbf16>
    %7628 = "ttir.permute"(%7626, %7627) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %7629 = ttir.empty() : tensor<640x3072xbf16>
    %7630 = "ttir.reshape"(%7628, %7629) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7631 = ttir.empty() : tensor<1x3072x3072xbf16>
    %7632 = "ttir.reshape"(%arg246, %7631) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %7633 = ttir.empty() : tensor<3072x3072xbf16>
    %7634 = "ttir.reshape"(%7632, %7633) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7635 = ttir.empty() : tensor<3072x3072xbf16>
    %7636 = "ttir.permute"(%7634, %7635) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7637 = "ttir.dot_general"(%7630, %7636) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %7638 = ttir.empty() : tensor<1x640x3072xbf16>
    %7639 = "ttir.reshape"(%7637, %7638) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7640 = ttir.empty() : tensor<1x640x3072xbf16>
    %7641 = "ttir.add"(%7451, %7639, %7640) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7642 = ttir.empty() : tensor<1x1x3072xbf16>
    %7643 = "ttir.reshape"(%arg249, %7642) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %7644 = ttir.empty() : tensor<3072xbf16>
    %7645 = "ttir.reshape"(%7643, %7644) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %7646 = ttir.empty() : tensor<3072xf32>
    %7647 = "ttir.typecast"(%7645, %7646) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %7648 = ttir.empty() : tensor<1x1x3072xf32>
    %7649 = "ttir.reshape"(%7647, %7648) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %7650 = ttir.empty() : tensor<1x640x3072xf32>
    %7651 = "ttir.broadcast"(%7649, %7650) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7652 = ttir.empty() : tensor<1x640x3072xf32>
    %7653 = "ttir.typecast"(%7641, %7652) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7654 = ttir.empty() : tensor<1x640x3072xf32>
    %7655 = "ttir.pow"(%7653, %5, %7654) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7656 = ttir.empty() : tensor<1x640xf32>
    %7657 = "ttir.sum"(%7655, %7656) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7658 = ttir.empty() : tensor<1x640xf32>
    %7659 = "ttir.multiply"(%7657, %4, %7658) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7660 = ttir.empty() : tensor<1x640x1xf32>
    %7661 = "ttir.reshape"(%7659, %7660) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7662 = ttir.empty() : tensor<1x640x1xf32>
    %7663 = "ttir.add"(%7661, %46, %7662) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7664 = ttir.empty() : tensor<1x640x1xf32>
    %7665 = "ttir.rsqrt"(%7663, %7664) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7666 = ttir.empty() : tensor<1x640xf32>
    %7667 = "ttir.reshape"(%7665, %7666) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7668 = ttir.empty() : tensor<1x640x1xf32>
    %7669 = "ttir.reshape"(%7667, %7668) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7670 = ttir.empty() : tensor<1x640x3072xf32>
    %7671 = "ttir.broadcast"(%7669, %7670) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7672 = ttir.empty() : tensor<1x640x3072xf32>
    %7673 = "ttir.multiply"(%7653, %7671, %7672) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7674 = ttir.empty() : tensor<1x640x3072xbf16>
    %7675 = "ttir.typecast"(%7673, %7674) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7676 = ttir.empty() : tensor<1x640x3072xf32>
    %7677 = "ttir.typecast"(%7675, %7676) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7678 = ttir.empty() : tensor<1x640x3072xf32>
    %7679 = "ttir.multiply"(%7651, %7677, %7678) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7680 = ttir.empty() : tensor<1x640x3072xbf16>
    %7681 = "ttir.typecast"(%7679, %7680) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7682 = ttir.empty() : tensor<640x3072xbf16>
    %7683 = "ttir.reshape"(%7681, %7682) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7684 = ttir.empty() : tensor<1x8192x3072xbf16>
    %7685 = "ttir.reshape"(%arg250, %7684) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %7686 = ttir.empty() : tensor<8192x3072xbf16>
    %7687 = "ttir.reshape"(%7685, %7686) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7688 = ttir.empty() : tensor<3072x8192xbf16>
    %7689 = "ttir.permute"(%7687, %7688) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7690 = "ttir.dot_general"(%7683, %7689) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %7691 = ttir.empty() : tensor<1x640x8192xbf16>
    %7692 = "ttir.reshape"(%7690, %7691) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7693 = ttir.empty() : tensor<1x640x8192xf32>
    %7694 = "ttir.typecast"(%7692, %7693) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7695 = ttir.empty() : tensor<1x640x8192xbf16>
    %7696 = "ttir.sigmoid"(%7692, %7695) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7697 = ttir.empty() : tensor<1x640x8192xf32>
    %7698 = "ttir.typecast"(%7696, %7697) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7699 = ttir.empty() : tensor<1x640x8192xf32>
    %7700 = "ttir.multiply"(%7694, %7698, %7699) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7701 = ttir.empty() : tensor<1x640x8192xbf16>
    %7702 = "ttir.typecast"(%7700, %7701) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7703 = ttir.empty() : tensor<1x640x8192xf32>
    %7704 = "ttir.typecast"(%7702, %7703) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7705 = ttir.empty() : tensor<1x8192x3072xbf16>
    %7706 = "ttir.reshape"(%arg245, %7705) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %7707 = ttir.empty() : tensor<8192x3072xbf16>
    %7708 = "ttir.reshape"(%7706, %7707) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7709 = ttir.empty() : tensor<3072x8192xbf16>
    %7710 = "ttir.permute"(%7708, %7709) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7711 = "ttir.dot_general"(%7683, %7710) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %7712 = ttir.empty() : tensor<1x640x8192xbf16>
    %7713 = "ttir.reshape"(%7711, %7712) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7714 = ttir.empty() : tensor<1x640x8192xf32>
    %7715 = "ttir.typecast"(%7713, %7714) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7716 = ttir.empty() : tensor<1x640x8192xf32>
    %7717 = "ttir.multiply"(%7704, %7715, %7716) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7718 = ttir.empty() : tensor<1x640x8192xbf16>
    %7719 = "ttir.typecast"(%7717, %7718) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7720 = ttir.empty() : tensor<640x8192xbf16>
    %7721 = "ttir.reshape"(%7719, %7720) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %7722 = ttir.empty() : tensor<1x3072x8192xbf16>
    %7723 = "ttir.reshape"(%arg244, %7722) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %7724 = ttir.empty() : tensor<3072x8192xbf16>
    %7725 = "ttir.reshape"(%7723, %7724) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7726 = ttir.empty() : tensor<8192x3072xbf16>
    %7727 = "ttir.permute"(%7725, %7726) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7728 = "ttir.dot_general"(%7721, %7727) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %7729 = ttir.empty() : tensor<1x640x3072xbf16>
    %7730 = "ttir.reshape"(%7728, %7729) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7731 = ttir.empty() : tensor<1x640x3072xbf16>
    %7732 = "ttir.add"(%7641, %7730, %7731) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7733 = ttir.empty() : tensor<1x640x3072xf32>
    %7734 = "ttir.typecast"(%7732, %7733) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7735 = ttir.empty() : tensor<1x640x3072xf32>
    %7736 = "ttir.pow"(%7734, %5, %7735) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7737 = ttir.empty() : tensor<1x640xf32>
    %7738 = "ttir.sum"(%7736, %7737) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7739 = ttir.empty() : tensor<1x640xf32>
    %7740 = "ttir.multiply"(%7738, %4, %7739) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7741 = ttir.empty() : tensor<1x640x1xf32>
    %7742 = "ttir.reshape"(%7740, %7741) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7743 = ttir.empty() : tensor<1x640x1xf32>
    %7744 = "ttir.add"(%7742, %46, %7743) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7745 = ttir.empty() : tensor<1x640x1xf32>
    %7746 = "ttir.rsqrt"(%7744, %7745) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7747 = ttir.empty() : tensor<1x640xf32>
    %7748 = "ttir.reshape"(%7746, %7747) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7749 = ttir.empty() : tensor<1x640x1xf32>
    %7750 = "ttir.reshape"(%7748, %7749) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7751 = ttir.empty() : tensor<1x640x3072xf32>
    %7752 = "ttir.broadcast"(%7750, %7751) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7753 = ttir.empty() : tensor<1x640x3072xf32>
    %7754 = "ttir.multiply"(%7734, %7752, %7753) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7755 = ttir.empty() : tensor<1x640x3072xbf16>
    %7756 = "ttir.typecast"(%7754, %7755) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7757 = ttir.empty() : tensor<1x640x3072xf32>
    %7758 = "ttir.typecast"(%7756, %7757) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7759 = ttir.empty() : tensor<1x640x3072xf32>
    %7760 = "ttir.multiply"(%7504, %7758, %7759) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7761 = ttir.empty() : tensor<1x640x3072xbf16>
    %7762 = "ttir.typecast"(%7760, %7761) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7763 = ttir.empty() : tensor<640x3072xbf16>
    %7764 = "ttir.reshape"(%7762, %7763) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7765 = ttir.empty() : tensor<1x1024x3072xbf16>
    %7766 = "ttir.reshape"(%arg243, %7765) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %7767 = ttir.empty() : tensor<1024x3072xbf16>
    %7768 = "ttir.reshape"(%7766, %7767) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %7769 = ttir.empty() : tensor<3072x1024xbf16>
    %7770 = "ttir.permute"(%7768, %7769) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %7771 = "ttir.dot_general"(%7764, %7770) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %7772 = ttir.empty() : tensor<1x640x8x128xbf16>
    %7773 = "ttir.reshape"(%7771, %7772) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %7774 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7775 = "ttir.permute"(%7773, %7774) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7776 = ttir.empty() : tensor<1x1024x3072xbf16>
    %7777 = "ttir.reshape"(%arg252, %7776) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %7778 = ttir.empty() : tensor<1024x3072xbf16>
    %7779 = "ttir.reshape"(%7777, %7778) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %7780 = ttir.empty() : tensor<3072x1024xbf16>
    %7781 = "ttir.permute"(%7779, %7780) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %7782 = "ttir.dot_general"(%7764, %7781) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<640x1024xbf16>
    %7783 = ttir.empty() : tensor<1x640x8x128xbf16>
    %7784 = "ttir.reshape"(%7782, %7783) <{shape = [1 : i32, 640 : i32, 8 : i32, 128 : i32]}> : (tensor<640x1024xbf16>, tensor<1x640x8x128xbf16>) -> tensor<1x640x8x128xbf16>
    %7785 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7786 = "ttir.permute"(%7784, %7785) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x8x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7787 = ttir.empty() : tensor<1x8x640x128xf32>
    %7788 = "ttir.typecast"(%7786, %7787) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7789 = ttir.empty() : tensor<1x8x640x128xf32>
    %7790 = "ttir.multiply"(%7788, %178, %7789) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7791 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7792 = "ttir.typecast"(%7790, %7791) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7793 = ttir.empty() : tensor<1x8x640x64xbf16>
    %7794 = "ttir.slice_static"(%7786, %7793) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %7795 = ttir.empty() : tensor<1x8x640x64xbf16>
    %7796 = "ttir.neg"(%7794, %7795) : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %7797 = ttir.empty() : tensor<1x8x640x64xbf16>
    %7798 = "ttir.slice_static"(%7786, %7797) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x64xbf16>) -> tensor<1x8x640x64xbf16>
    %7799 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7800 = "ttir.concat"(%7796, %7798, %7799) <{dim = 3 : si32}> : (tensor<1x8x640x64xbf16>, tensor<1x8x640x64xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7801 = ttir.empty() : tensor<1x8x640x128xf32>
    %7802 = "ttir.typecast"(%7800, %7801) <{conservative_folding = false}> : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7803 = ttir.empty() : tensor<1x8x640x128xf32>
    %7804 = "ttir.multiply"(%7802, %196, %7803) : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>, tensor<1x8x640x128xf32>) -> tensor<1x8x640x128xf32>
    %7805 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7806 = "ttir.typecast"(%7804, %7805) <{conservative_folding = false}> : (tensor<1x8x640x128xf32>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7807 = ttir.empty() : tensor<1x8x640x128xbf16>
    %7808 = "ttir.add"(%7792, %7806, %7807) : (tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>) -> tensor<1x8x640x128xbf16>
    %7809 = ttir.empty() : tensor<1x1x3072xbf16>
    %7810 = "ttir.reshape"(%arg260, %7809) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %7811 = ttir.empty() : tensor<3072xbf16>
    %7812 = "ttir.reshape"(%7810, %7811) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %7813 = ttir.empty() : tensor<3072xf32>
    %7814 = "ttir.typecast"(%7812, %7813) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %7815 = ttir.empty() : tensor<1x1x3072xf32>
    %7816 = "ttir.reshape"(%7814, %7815) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %7817 = ttir.empty() : tensor<1x640x3072xf32>
    %7818 = "ttir.broadcast"(%7816, %7817) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7819 = ttir.empty() : tensor<1x3072x3072xbf16>
    %7820 = "ttir.reshape"(%arg257, %7819) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %7821 = ttir.empty() : tensor<3072x3072xbf16>
    %7822 = "ttir.reshape"(%7820, %7821) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7823 = ttir.empty() : tensor<3072x3072xbf16>
    %7824 = "ttir.permute"(%7822, %7823) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7825 = "ttir.dot_general"(%7764, %7824) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %7826 = ttir.empty() : tensor<1x640x24x128xbf16>
    %7827 = "ttir.reshape"(%7825, %7826) <{shape = [1 : i32, 640 : i32, 24 : i32, 128 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %7828 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7829 = "ttir.permute"(%7827, %7828) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x640x24x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7830 = ttir.empty() : tensor<1x24x640x128xf32>
    %7831 = "ttir.typecast"(%7829, %7830) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7832 = ttir.empty() : tensor<1x24x640x128xf32>
    %7833 = "ttir.multiply"(%7831, %125, %7832) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7834 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7835 = "ttir.typecast"(%7833, %7834) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7836 = ttir.empty() : tensor<1x24x640x64xbf16>
    %7837 = "ttir.slice_static"(%7829, %7836) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %7838 = ttir.empty() : tensor<1x24x640x64xbf16>
    %7839 = "ttir.neg"(%7837, %7838) : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %7840 = ttir.empty() : tensor<1x24x640x64xbf16>
    %7841 = "ttir.slice_static"(%7829, %7840) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 24 : i32, 640 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x64xbf16>) -> tensor<1x24x640x64xbf16>
    %7842 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7843 = "ttir.concat"(%7839, %7841, %7842) <{dim = 3 : si32}> : (tensor<1x24x640x64xbf16>, tensor<1x24x640x64xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7844 = ttir.empty() : tensor<1x24x640x128xf32>
    %7845 = "ttir.typecast"(%7843, %7844) <{conservative_folding = false}> : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7846 = ttir.empty() : tensor<1x24x640x128xf32>
    %7847 = "ttir.multiply"(%7845, %153, %7846) : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>, tensor<1x24x640x128xf32>) -> tensor<1x24x640x128xf32>
    %7848 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7849 = "ttir.typecast"(%7847, %7848) <{conservative_folding = false}> : (tensor<1x24x640x128xf32>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7850 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7851 = "ttir.add"(%7835, %7849, %7850) : (tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7852 = ttir.empty() : tensor<24x640x128xbf16>
    %7853 = "ttir.reshape"(%7851, %7852) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x24x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7854 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %7855 = "ttir.reshape"(%7808, %7854) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %7856 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %7857 = "ttir.broadcast"(%7855, %7856) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %7858 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7859 = "ttir.reshape"(%7857, %7858) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7860 = ttir.empty() : tensor<1x24x128x640xbf16>
    %7861 = "ttir.permute"(%7859, %7860) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x24x640x128xbf16>, tensor<1x24x128x640xbf16>) -> tensor<1x24x128x640xbf16>
    %7862 = ttir.empty() : tensor<24x128x640xbf16>
    %7863 = "ttir.reshape"(%7861, %7862) <{shape = [24 : i32, 128 : i32, 640 : i32]}> : (tensor<1x24x128x640xbf16>, tensor<24x128x640xbf16>) -> tensor<24x128x640xbf16>
    %7864 = "ttir.dot_general"(%7853, %7863) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x128xbf16>, tensor<24x128x640xbf16>) -> tensor<24x640x640xbf16>
    %7865 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7866 = "ttir.reshape"(%7864, %7865) <{shape = [1 : i32, 24 : i32, 640 : i32, 640 : i32]}> : (tensor<24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7867 = ttir.empty() : tensor<1x24x640x640xf32>
    %7868 = "ttir.typecast"(%7866, %7867) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7869 = ttir.empty() : tensor<1x24x640x640xf32>
    %7870 = "ttir.multiply"(%7868, %221, %7869) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7871 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7872 = "ttir.typecast"(%7870, %7871) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7873 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7874 = "ttir.add"(%7872, %285, %7873) : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7875 = ttir.empty() : tensor<1x24x640x640xf32>
    %7876 = "ttir.typecast"(%7874, %7875) <{conservative_folding = false}> : (tensor<1x24x640x640xbf16>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7877 = ttir.empty() : tensor<1x24x640xf32>
    %7878 = "ttir.max"(%7876, %7877) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %7879 = ttir.empty() : tensor<1x24x640x1xf32>
    %7880 = "ttir.reshape"(%7878, %7879) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %7881 = ttir.empty() : tensor<1x24x640x640xf32>
    %7882 = "ttir.broadcast"(%7880, %7881) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7883 = ttir.empty() : tensor<1x24x640x640xf32>
    %7884 = "ttir.subtract"(%7876, %7882, %7883) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7885 = ttir.empty() : tensor<1x24x640x640xf32>
    %7886 = "ttir.exp"(%7884, %7885) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7887 = ttir.empty() : tensor<1x24x640xf32>
    %7888 = "ttir.sum"(%7886, %7887) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640xf32>) -> tensor<1x24x640xf32>
    %7889 = ttir.empty() : tensor<1x24x640x1xf32>
    %7890 = "ttir.reshape"(%7888, %7889) <{shape = [1 : i32, 24 : i32, 640 : i32, 1 : i32]}> : (tensor<1x24x640xf32>, tensor<1x24x640x1xf32>) -> tensor<1x24x640x1xf32>
    %7891 = ttir.empty() : tensor<1x24x640x640xf32>
    %7892 = "ttir.broadcast"(%7890, %7891) <{broadcast_dimensions = array<i64: 1, 1, 1, 640>}> : (tensor<1x24x640x1xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7893 = ttir.empty() : tensor<1x24x640x640xf32>
    %7894 = "ttir.div"(%7886, %7892, %7893) : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>, tensor<1x24x640x640xf32>) -> tensor<1x24x640x640xf32>
    %7895 = ttir.empty() : tensor<1x24x640x640xbf16>
    %7896 = "ttir.typecast"(%7894, %7895) <{conservative_folding = false}> : (tensor<1x24x640x640xf32>, tensor<1x24x640x640xbf16>) -> tensor<1x24x640x640xbf16>
    %7897 = ttir.empty() : tensor<24x640x640xbf16>
    %7898 = "ttir.reshape"(%7896, %7897) <{shape = [24 : i32, 640 : i32, 640 : i32]}> : (tensor<1x24x640x640xbf16>, tensor<24x640x640xbf16>) -> tensor<24x640x640xbf16>
    %7899 = ttir.empty() : tensor<1x8x1x640x128xbf16>
    %7900 = "ttir.reshape"(%7775, %7899) <{shape = [1 : i32, 8 : i32, 1 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x640x128xbf16>, tensor<1x8x1x640x128xbf16>) -> tensor<1x8x1x640x128xbf16>
    %7901 = ttir.empty() : tensor<1x8x3x640x128xbf16>
    %7902 = "ttir.broadcast"(%7900, %7901) <{broadcast_dimensions = array<i64: 1, 1, 3, 1, 1>}> : (tensor<1x8x1x640x128xbf16>, tensor<1x8x3x640x128xbf16>) -> tensor<1x8x3x640x128xbf16>
    %7903 = ttir.empty() : tensor<24x640x128xbf16>
    %7904 = "ttir.reshape"(%7902, %7903) <{shape = [24 : i32, 640 : i32, 128 : i32]}> : (tensor<1x8x3x640x128xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7905 = "ttir.dot_general"(%7898, %7904) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<24x640x640xbf16>, tensor<24x640x128xbf16>) -> tensor<24x640x128xbf16>
    %7906 = ttir.empty() : tensor<1x24x640x128xbf16>
    %7907 = "ttir.reshape"(%7905, %7906) <{shape = [1 : i32, 24 : i32, 640 : i32, 128 : i32]}> : (tensor<24x640x128xbf16>, tensor<1x24x640x128xbf16>) -> tensor<1x24x640x128xbf16>
    %7908 = ttir.empty() : tensor<1x640x24x128xbf16>
    %7909 = "ttir.permute"(%7907, %7908) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x640x128xbf16>, tensor<1x640x24x128xbf16>) -> tensor<1x640x24x128xbf16>
    %7910 = ttir.empty() : tensor<640x3072xbf16>
    %7911 = "ttir.reshape"(%7909, %7910) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x24x128xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7912 = ttir.empty() : tensor<1x3072x3072xbf16>
    %7913 = "ttir.reshape"(%arg256, %7912) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %7914 = ttir.empty() : tensor<3072x3072xbf16>
    %7915 = "ttir.reshape"(%7913, %7914) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7916 = ttir.empty() : tensor<3072x3072xbf16>
    %7917 = "ttir.permute"(%7915, %7916) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %7918 = "ttir.dot_general"(%7911, %7917) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<640x3072xbf16>
    %7919 = ttir.empty() : tensor<1x640x3072xbf16>
    %7920 = "ttir.reshape"(%7918, %7919) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7921 = ttir.empty() : tensor<1x640x3072xbf16>
    %7922 = "ttir.add"(%7732, %7920, %7921) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7923 = ttir.empty() : tensor<1x1x3072xbf16>
    %7924 = "ttir.reshape"(%arg258, %7923) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %7925 = ttir.empty() : tensor<3072xbf16>
    %7926 = "ttir.reshape"(%7924, %7925) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<3072xbf16>
    %7927 = ttir.empty() : tensor<3072xf32>
    %7928 = "ttir.typecast"(%7926, %7927) <{conservative_folding = false}> : (tensor<3072xbf16>, tensor<3072xf32>) -> tensor<3072xf32>
    %7929 = ttir.empty() : tensor<1x1x3072xf32>
    %7930 = "ttir.reshape"(%7928, %7929) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    %7931 = ttir.empty() : tensor<1x640x3072xf32>
    %7932 = "ttir.broadcast"(%7930, %7931) <{broadcast_dimensions = array<i64: 1, 640, 1>}> : (tensor<1x1x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7933 = ttir.empty() : tensor<1x640x3072xf32>
    %7934 = "ttir.typecast"(%7922, %7933) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7935 = ttir.empty() : tensor<1x640x3072xf32>
    %7936 = "ttir.pow"(%7934, %5, %7935) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7937 = ttir.empty() : tensor<1x640xf32>
    %7938 = "ttir.sum"(%7936, %7937) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7939 = ttir.empty() : tensor<1x640xf32>
    %7940 = "ttir.multiply"(%7938, %4, %7939) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7941 = ttir.empty() : tensor<1x640x1xf32>
    %7942 = "ttir.reshape"(%7940, %7941) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7943 = ttir.empty() : tensor<1x640x1xf32>
    %7944 = "ttir.add"(%7942, %46, %7943) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7945 = ttir.empty() : tensor<1x640x1xf32>
    %7946 = "ttir.rsqrt"(%7944, %7945) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7947 = ttir.empty() : tensor<1x640xf32>
    %7948 = "ttir.reshape"(%7946, %7947) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %7949 = ttir.empty() : tensor<1x640x1xf32>
    %7950 = "ttir.reshape"(%7948, %7949) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %7951 = ttir.empty() : tensor<1x640x3072xf32>
    %7952 = "ttir.broadcast"(%7950, %7951) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7953 = ttir.empty() : tensor<1x640x3072xf32>
    %7954 = "ttir.multiply"(%7934, %7952, %7953) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7955 = ttir.empty() : tensor<1x640x3072xbf16>
    %7956 = "ttir.typecast"(%7954, %7955) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7957 = ttir.empty() : tensor<1x640x3072xf32>
    %7958 = "ttir.typecast"(%7956, %7957) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7959 = ttir.empty() : tensor<1x640x3072xf32>
    %7960 = "ttir.multiply"(%7932, %7958, %7959) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %7961 = ttir.empty() : tensor<1x640x3072xbf16>
    %7962 = "ttir.typecast"(%7960, %7961) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %7963 = ttir.empty() : tensor<640x3072xbf16>
    %7964 = "ttir.reshape"(%7962, %7963) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %7965 = ttir.empty() : tensor<1x8192x3072xbf16>
    %7966 = "ttir.reshape"(%arg259, %7965) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %7967 = ttir.empty() : tensor<8192x3072xbf16>
    %7968 = "ttir.reshape"(%7966, %7967) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7969 = ttir.empty() : tensor<3072x8192xbf16>
    %7970 = "ttir.permute"(%7968, %7969) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7971 = "ttir.dot_general"(%7964, %7970) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %7972 = ttir.empty() : tensor<1x640x8192xbf16>
    %7973 = "ttir.reshape"(%7971, %7972) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7974 = ttir.empty() : tensor<1x640x8192xf32>
    %7975 = "ttir.typecast"(%7973, %7974) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7976 = ttir.empty() : tensor<1x640x8192xbf16>
    %7977 = "ttir.sigmoid"(%7973, %7976) : (tensor<1x640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7978 = ttir.empty() : tensor<1x640x8192xf32>
    %7979 = "ttir.typecast"(%7977, %7978) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7980 = ttir.empty() : tensor<1x640x8192xf32>
    %7981 = "ttir.multiply"(%7975, %7979, %7980) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7982 = ttir.empty() : tensor<1x640x8192xbf16>
    %7983 = "ttir.typecast"(%7981, %7982) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7984 = ttir.empty() : tensor<1x640x8192xf32>
    %7985 = "ttir.typecast"(%7983, %7984) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7986 = ttir.empty() : tensor<1x8192x3072xbf16>
    %7987 = "ttir.reshape"(%arg255, %7986) <{shape = [1 : i32, 8192 : i32, 3072 : i32]}> : (tensor<8192x3072xbf16>, tensor<1x8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %7988 = ttir.empty() : tensor<8192x3072xbf16>
    %7989 = "ttir.reshape"(%7987, %7988) <{shape = [8192 : i32, 3072 : i32]}> : (tensor<1x8192x3072xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %7990 = ttir.empty() : tensor<3072x8192xbf16>
    %7991 = "ttir.permute"(%7989, %7990) <{permutation = array<i64: 1, 0>}> : (tensor<8192x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %7992 = "ttir.dot_general"(%7964, %7991) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<640x8192xbf16>
    %7993 = ttir.empty() : tensor<1x640x8192xbf16>
    %7994 = "ttir.reshape"(%7992, %7993) <{shape = [1 : i32, 640 : i32, 8192 : i32]}> : (tensor<640x8192xbf16>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %7995 = ttir.empty() : tensor<1x640x8192xf32>
    %7996 = "ttir.typecast"(%7994, %7995) <{conservative_folding = false}> : (tensor<1x640x8192xbf16>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7997 = ttir.empty() : tensor<1x640x8192xf32>
    %7998 = "ttir.multiply"(%7985, %7996, %7997) : (tensor<1x640x8192xf32>, tensor<1x640x8192xf32>, tensor<1x640x8192xf32>) -> tensor<1x640x8192xf32>
    %7999 = ttir.empty() : tensor<1x640x8192xbf16>
    %8000 = "ttir.typecast"(%7998, %7999) <{conservative_folding = false}> : (tensor<1x640x8192xf32>, tensor<1x640x8192xbf16>) -> tensor<1x640x8192xbf16>
    %8001 = ttir.empty() : tensor<640x8192xbf16>
    %8002 = "ttir.reshape"(%8000, %8001) <{shape = [640 : i32, 8192 : i32]}> : (tensor<1x640x8192xbf16>, tensor<640x8192xbf16>) -> tensor<640x8192xbf16>
    %8003 = ttir.empty() : tensor<1x3072x8192xbf16>
    %8004 = "ttir.reshape"(%arg254, %8003) <{shape = [1 : i32, 3072 : i32, 8192 : i32]}> : (tensor<3072x8192xbf16>, tensor<1x3072x8192xbf16>) -> tensor<1x3072x8192xbf16>
    %8005 = ttir.empty() : tensor<3072x8192xbf16>
    %8006 = "ttir.reshape"(%8004, %8005) <{shape = [3072 : i32, 8192 : i32]}> : (tensor<1x3072x8192xbf16>, tensor<3072x8192xbf16>) -> tensor<3072x8192xbf16>
    %8007 = ttir.empty() : tensor<8192x3072xbf16>
    %8008 = "ttir.permute"(%8006, %8007) <{permutation = array<i64: 1, 0>}> : (tensor<3072x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %8009 = "ttir.dot_general"(%8002, %8008) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x8192xbf16>, tensor<8192x3072xbf16>) -> tensor<640x3072xbf16>
    %8010 = ttir.empty() : tensor<1x640x3072xbf16>
    %8011 = "ttir.reshape"(%8009, %8010) <{shape = [1 : i32, 640 : i32, 3072 : i32]}> : (tensor<640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %8012 = ttir.empty() : tensor<1x640x3072xbf16>
    %8013 = "ttir.add"(%7922, %8011, %8012) : (tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %8014 = ttir.empty() : tensor<1x640x3072xf32>
    %8015 = "ttir.typecast"(%8013, %8014) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %8016 = ttir.empty() : tensor<1x640x3072xf32>
    %8017 = "ttir.pow"(%8015, %5, %8016) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %8018 = ttir.empty() : tensor<1x640xf32>
    %8019 = "ttir.sum"(%8017, %8018) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x640x3072xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %8020 = ttir.empty() : tensor<1x640xf32>
    %8021 = "ttir.multiply"(%8019, %4, %8020) : (tensor<1x640xf32>, tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %8022 = ttir.empty() : tensor<1x640x1xf32>
    %8023 = "ttir.reshape"(%8021, %8022) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %8024 = ttir.empty() : tensor<1x640x1xf32>
    %8025 = "ttir.add"(%8023, %46, %8024) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %8026 = ttir.empty() : tensor<1x640x1xf32>
    %8027 = "ttir.rsqrt"(%8025, %8026) : (tensor<1x640x1xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %8028 = ttir.empty() : tensor<1x640xf32>
    %8029 = "ttir.reshape"(%8027, %8028) <{shape = [1 : i32, 640 : i32]}> : (tensor<1x640x1xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %8030 = ttir.empty() : tensor<1x640x1xf32>
    %8031 = "ttir.reshape"(%8029, %8030) <{shape = [1 : i32, 640 : i32, 1 : i32]}> : (tensor<1x640xf32>, tensor<1x640x1xf32>) -> tensor<1x640x1xf32>
    %8032 = ttir.empty() : tensor<1x640x3072xf32>
    %8033 = "ttir.broadcast"(%8031, %8032) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x640x1xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %8034 = ttir.empty() : tensor<1x640x3072xf32>
    %8035 = "ttir.multiply"(%8015, %8033, %8034) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %8036 = ttir.empty() : tensor<1x640x3072xbf16>
    %8037 = "ttir.typecast"(%8035, %8036) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %8038 = ttir.empty() : tensor<1x640x3072xf32>
    %8039 = "ttir.typecast"(%8037, %8038) <{conservative_folding = false}> : (tensor<1x640x3072xbf16>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %8040 = ttir.empty() : tensor<1x640x3072xf32>
    %8041 = "ttir.multiply"(%7818, %8039, %8040) : (tensor<1x640x3072xf32>, tensor<1x640x3072xf32>, tensor<1x640x3072xf32>) -> tensor<1x640x3072xf32>
    %8042 = ttir.empty() : tensor<1x640x3072xbf16>
    %8043 = "ttir.typecast"(%8041, %8042) <{conservative_folding = false}> : (tensor<1x640x3072xf32>, tensor<1x640x3072xbf16>) -> tensor<1x640x3072xbf16>
    %8044 = ttir.empty() : tensor<640x3072xbf16>
    %8045 = "ttir.reshape"(%8043, %8044) <{shape = [640 : i32, 3072 : i32]}> : (tensor<1x640x3072xbf16>, tensor<640x3072xbf16>) -> tensor<640x3072xbf16>
    %8046 = ttir.empty() : tensor<1x128256x3072xbf16>
    %8047 = "ttir.reshape"(%arg253, %8046) <{shape = [1 : i32, 128256 : i32, 3072 : i32]}> : (tensor<128256x3072xbf16>, tensor<1x128256x3072xbf16>) -> tensor<1x128256x3072xbf16>
    %8048 = ttir.empty() : tensor<128256x3072xbf16>
    %8049 = "ttir.reshape"(%8047, %8048) <{shape = [128256 : i32, 3072 : i32]}> : (tensor<1x128256x3072xbf16>, tensor<128256x3072xbf16>) -> tensor<128256x3072xbf16>
    %8050 = ttir.empty() : tensor<3072x128256xbf16>
    %8051 = "ttir.permute"(%8049, %8050) <{permutation = array<i64: 1, 0>}> : (tensor<128256x3072xbf16>, tensor<3072x128256xbf16>) -> tensor<3072x128256xbf16>
    %8052 = "ttir.dot_general"(%8045, %8051) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<640x3072xbf16>, tensor<3072x128256xbf16>) -> tensor<640x128256xbf16>
    %8053 = ttir.empty() : tensor<1x640x128256xbf16>
    %8054 = "ttir.reshape"(%8052, %8053) <{shape = [1 : i32, 640 : i32, 128256 : i32]}> : (tensor<640x128256xbf16>, tensor<1x640x128256xbf16>) -> tensor<1x640x128256xbf16>
    return %79, %469, %750, %1031, %1312, %1593, %1874, %2155, %2436, %2717, %2998, %3279, %3560, %3841, %4122, %4403, %4684, %4965, %5246, %5527, %5808, %6089, %6370, %6651, %6932, %7213, %7494, %7775, %202, %547, %828, %1109, %1390, %1671, %1952, %2233, %2514, %2795, %3076, %3357, %3638, %3919, %4200, %4481, %4762, %5043, %5324, %5605, %5886, %6167, %6448, %6729, %7010, %7291, %7572, %7808, %8054 : tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x8x640x128xbf16>, tensor<1x640x128256xbf16>
  }
}
