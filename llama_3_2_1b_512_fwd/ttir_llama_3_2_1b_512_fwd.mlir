module @llama_3_2_1b_512_fwd attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @llama_3_2_1b_512_fwd attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @main(%arg0: tensor<128256x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128256x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.embed_tokens.weight"}, %arg1: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.q_proj.weight"}, %arg2: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.k_proj.weight"}, %arg3: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.v_proj.weight"}, %arg4: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.o_proj.weight"}, %arg5: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.mlp.gate_proj.weight"}, %arg6: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.mlp.up_proj.weight"}, %arg7: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.mlp.down_proj.weight"}, %arg8: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.input_layernorm.weight"}, %arg9: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.post_attention_layernorm.weight"}, %arg10: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.q_proj.weight"}, %arg11: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.k_proj.weight"}, %arg12: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.v_proj.weight"}, %arg13: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.o_proj.weight"}, %arg14: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.mlp.gate_proj.weight"}, %arg15: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.mlp.up_proj.weight"}, %arg16: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.mlp.down_proj.weight"}, %arg17: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.input_layernorm.weight"}, %arg18: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.post_attention_layernorm.weight"}, %arg19: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.q_proj.weight"}, %arg20: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.k_proj.weight"}, %arg21: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.v_proj.weight"}, %arg22: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.o_proj.weight"}, %arg23: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.mlp.gate_proj.weight"}, %arg24: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.mlp.up_proj.weight"}, %arg25: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.mlp.down_proj.weight"}, %arg26: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.input_layernorm.weight"}, %arg27: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.post_attention_layernorm.weight"}, %arg28: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.q_proj.weight"}, %arg29: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.k_proj.weight"}, %arg30: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.v_proj.weight"}, %arg31: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.o_proj.weight"}, %arg32: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.mlp.gate_proj.weight"}, %arg33: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.mlp.up_proj.weight"}, %arg34: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.mlp.down_proj.weight"}, %arg35: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.input_layernorm.weight"}, %arg36: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.post_attention_layernorm.weight"}, %arg37: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.q_proj.weight"}, %arg38: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.k_proj.weight"}, %arg39: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.v_proj.weight"}, %arg40: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.o_proj.weight"}, %arg41: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.mlp.gate_proj.weight"}, %arg42: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.mlp.up_proj.weight"}, %arg43: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.mlp.down_proj.weight"}, %arg44: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.input_layernorm.weight"}, %arg45: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.post_attention_layernorm.weight"}, %arg46: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.q_proj.weight"}, %arg47: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.k_proj.weight"}, %arg48: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.v_proj.weight"}, %arg49: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.o_proj.weight"}, %arg50: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.mlp.gate_proj.weight"}, %arg51: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.mlp.up_proj.weight"}, %arg52: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.mlp.down_proj.weight"}, %arg53: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.input_layernorm.weight"}, %arg54: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.post_attention_layernorm.weight"}, %arg55: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.q_proj.weight"}, %arg56: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.k_proj.weight"}, %arg57: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.v_proj.weight"}, %arg58: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.o_proj.weight"}, %arg59: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.mlp.gate_proj.weight"}, %arg60: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.mlp.up_proj.weight"}, %arg61: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.mlp.down_proj.weight"}, %arg62: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.input_layernorm.weight"}, %arg63: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.post_attention_layernorm.weight"}, %arg64: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.q_proj.weight"}, %arg65: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.k_proj.weight"}, %arg66: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.v_proj.weight"}, %arg67: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.o_proj.weight"}, %arg68: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.mlp.gate_proj.weight"}, %arg69: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.mlp.up_proj.weight"}, %arg70: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.mlp.down_proj.weight"}, %arg71: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.input_layernorm.weight"}, %arg72: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.post_attention_layernorm.weight"}, %arg73: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.q_proj.weight"}, %arg74: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.k_proj.weight"}, %arg75: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.v_proj.weight"}, %arg76: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.o_proj.weight"}, %arg77: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.mlp.gate_proj.weight"}, %arg78: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.mlp.up_proj.weight"}, %arg79: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.mlp.down_proj.weight"}, %arg80: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.input_layernorm.weight"}, %arg81: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.post_attention_layernorm.weight"}, %arg82: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.q_proj.weight"}, %arg83: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.k_proj.weight"}, %arg84: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.v_proj.weight"}, %arg85: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.o_proj.weight"}, %arg86: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.mlp.gate_proj.weight"}, %arg87: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.mlp.up_proj.weight"}, %arg88: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.mlp.down_proj.weight"}, %arg89: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.input_layernorm.weight"}, %arg90: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.post_attention_layernorm.weight"}, %arg91: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.q_proj.weight"}, %arg92: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.k_proj.weight"}, %arg93: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.v_proj.weight"}, %arg94: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.o_proj.weight"}, %arg95: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.mlp.gate_proj.weight"}, %arg96: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.mlp.up_proj.weight"}, %arg97: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.mlp.down_proj.weight"}, %arg98: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.input_layernorm.weight"}, %arg99: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.post_attention_layernorm.weight"}, %arg100: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.q_proj.weight"}, %arg101: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.k_proj.weight"}, %arg102: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.v_proj.weight"}, %arg103: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.o_proj.weight"}, %arg104: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.mlp.gate_proj.weight"}, %arg105: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.mlp.up_proj.weight"}, %arg106: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.mlp.down_proj.weight"}, %arg107: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.input_layernorm.weight"}, %arg108: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.post_attention_layernorm.weight"}, %arg109: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.q_proj.weight"}, %arg110: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.k_proj.weight"}, %arg111: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.v_proj.weight"}, %arg112: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.o_proj.weight"}, %arg113: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.mlp.gate_proj.weight"}, %arg114: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.mlp.up_proj.weight"}, %arg115: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.mlp.down_proj.weight"}, %arg116: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.input_layernorm.weight"}, %arg117: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.post_attention_layernorm.weight"}, %arg118: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.q_proj.weight"}, %arg119: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.k_proj.weight"}, %arg120: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.v_proj.weight"}, %arg121: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.o_proj.weight"}, %arg122: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.mlp.gate_proj.weight"}, %arg123: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.mlp.up_proj.weight"}, %arg124: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.mlp.down_proj.weight"}, %arg125: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.input_layernorm.weight"}, %arg126: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.post_attention_layernorm.weight"}, %arg127: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.q_proj.weight"}, %arg128: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.k_proj.weight"}, %arg129: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.v_proj.weight"}, %arg130: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.o_proj.weight"}, %arg131: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.mlp.gate_proj.weight"}, %arg132: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.mlp.up_proj.weight"}, %arg133: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.mlp.down_proj.weight"}, %arg134: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.input_layernorm.weight"}, %arg135: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.post_attention_layernorm.weight"}, %arg136: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.q_proj.weight"}, %arg137: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.k_proj.weight"}, %arg138: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.v_proj.weight"}, %arg139: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.o_proj.weight"}, %arg140: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.mlp.gate_proj.weight"}, %arg141: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.mlp.up_proj.weight"}, %arg142: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.mlp.down_proj.weight"}, %arg143: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.input_layernorm.weight"}, %arg144: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.post_attention_layernorm.weight"}, %arg145: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.norm.weight"}, %arg146: tensor<32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32xf32>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.rotary_emb.inv_freq"}, %arg147: tensor<1x512xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x512xi64>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "input_ids"}, %arg148: tensor<1x512xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x512xi64>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "attention_mask"}, %arg149: tensor<1x511x128256xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x511x128256xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "labels_one_hot"}, %arg150: tensor<1x511x1xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x511x1xf32>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "loss_weight"}) -> (tensor<1x1x1xf32>, tensor<128256x2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<1x512xi64>, tensor<1x511x128256xbf16>, tensor<1x511x1xf32>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x511x128256xbf16>, tensor<1x511x1xf32>, tensor<1x511x1xf32>) {
        %0 = "ttir.embedding"(%arg147, %arg0) : (tensor<1x512xi64>, tensor<128256x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 512 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<512xi64>
        %2 = "ttir.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
        %3 = "ttir.add"(%1, %2) : (tensor<512xi64>, tensor<1xi64>) -> tensor<512xi64>
        %4 = "ttir.unsqueeze"(%3) <{dim = 0 : si32}> : (tensor<512xi64>) -> tensor<1x512xi64>
        %5 = "ttir.typecast"(%arg148) <{conservative_folding = false}> : (tensor<1x512xi64>) -> tensor<1x512xi1>
        %6 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 1 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1xi64>
        %7 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 512 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<512xi64>
        %8 = "ttir.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
        %9 = "ttir.add"(%7, %8) : (tensor<512xi64>, tensor<1xi64>) -> tensor<512xi64>
        %10 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 512 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<512xi64>
        %11 = "ttir.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
        %12 = "ttir.add"(%10, %11) : (tensor<512xi64>, tensor<1xi64>) -> tensor<512xi64>
        %13 = "ttir.unsqueeze"(%6) <{dim = 1 : si32}> : (tensor<1xi64>) -> tensor<1x1xi64>
        %14 = "ttir.unsqueeze"(%13) <{dim = 2 : si32}> : (tensor<1x1xi64>) -> tensor<1x1x1xi64>
        %15 = "ttir.unsqueeze"(%14) <{dim = 3 : si32}> : (tensor<1x1x1xi64>) -> tensor<1x1x1x1xi64>
        %16 = "ttir.unsqueeze"(%9) <{dim = 0 : si32}> : (tensor<512xi64>) -> tensor<1x512xi64>
        %17 = "ttir.unsqueeze"(%16) <{dim = 1 : si32}> : (tensor<1x512xi64>) -> tensor<1x1x512xi64>
        %18 = "ttir.unsqueeze"(%17) <{dim = 3 : si32}> : (tensor<1x1x512xi64>) -> tensor<1x1x512x1xi64>
        %19 = "ttir.unsqueeze"(%12) <{dim = 0 : si32}> : (tensor<512xi64>) -> tensor<1x512xi64>
        %20 = "ttir.unsqueeze"(%19) <{dim = 1 : si32}> : (tensor<1x512xi64>) -> tensor<1x1x512xi64>
        %21 = "ttir.unsqueeze"(%20) <{dim = 2 : si32}> : (tensor<1x1x512xi64>) -> tensor<1x1x1x512xi64>
        %22 = "ttir.ones"() <{shape = array<i32>}> : () -> tensor<i1>
        %23 = "ttir.le"(%21, %18) : (tensor<1x1x1x512xi64>, tensor<1x1x512x1xi64>) -> tensor<1x1x512x512xi1>
        %24 = "ttir.logical_and"(%22, %23) : (tensor<i1>, tensor<1x1x512x512xi1>) -> tensor<1x1x512x512xi1>
        %25 = "ttir.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
        %26 = "ttir.constant"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
        %27 = "ttir.add"(%15, %26) : (tensor<1x1x1x1xi64>, tensor<1xi64>) -> tensor<1x1x1x1xi64>
        %28 = "ttir.lt"(%15, %25) : (tensor<1x1x1x1xi64>, tensor<1xi64>) -> tensor<1x1x1x1xi1>
        %29 = "ttir.where"(%28, %27, %15) : (tensor<1x1x1x1xi1>, tensor<1x1x1x1xi64>, tensor<1x1x1x1xi64>) -> tensor<1x1x1x1xi64>
        %30 = "ttir.broadcast"(%29) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x1x1x1xi64>) -> tensor<1x1x1x512xi64>
        %31 = "ttir.constant"() <{value = dense<512> : tensor<1xi64>}> : () -> tensor<1xi64>
        %32 = "ttir.multiply"(%30, %31) : (tensor<1x1x1x512xi64>, tensor<1xi64>) -> tensor<1x1x1x512xi64>
        %33 = "ttir.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
        %34 = "ttir.constant"() <{value = dense<512> : tensor<1xi64>}> : () -> tensor<1xi64>
        %35 = "ttir.add"(%21, %34) : (tensor<1x1x1x512xi64>, tensor<1xi64>) -> tensor<1x1x1x512xi64>
        %36 = "ttir.lt"(%21, %33) : (tensor<1x1x1x512xi64>, tensor<1xi64>) -> tensor<1x1x1x512xi1>
        %37 = "ttir.where"(%36, %35, %21) : (tensor<1x1x1x512xi1>, tensor<1x1x1x512xi64>, tensor<1x1x1x512xi64>) -> tensor<1x1x1x512xi64>
        %38 = "ttir.add"(%32, %37) : (tensor<1x1x1x512xi64>, tensor<1x1x1x512xi64>) -> tensor<1x1x1x512xi64>
        %39 = "ttir.reshape"(%5) <{shape = [512 : i32]}> : (tensor<1x512xi1>) -> tensor<512xi1>
        %40 = "ttir.typecast"(%38) <{conservative_folding = false}> : (tensor<1x1x1x512xi64>) -> tensor<1x1x1x512xi32>
        %41 = "ttir.reshape"(%40) <{shape = [512 : i32]}> : (tensor<1x1x1x512xi32>) -> tensor<512xi32>
        %42 = "ttir.gather"(%39, %41) <{dim = 0 : i32}> : (tensor<512xi1>, tensor<512xi32>) -> tensor<512xi1>
        %43 = "ttir.reshape"(%42) <{shape = [1 : i32, 1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xi1>) -> tensor<1x1x1x512xi1>
        %44 = "ttir.logical_and"(%24, %43) : (tensor<1x1x512x512xi1>, tensor<1x1x1x512xi1>) -> tensor<1x1x512x512xi1>
        %45 = "ttir.broadcast"(%44) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x512x512xi1>) -> tensor<1x1x512x512xi1>
        %46 = "ttir.unsqueeze"(%arg146) <{dim = 0 : si32}> : (tensor<32xf32>) -> tensor<1x32xf32>
        %47 = "ttir.unsqueeze"(%46) <{dim = 2 : si32}> : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
        %48 = "ttir.broadcast"(%47) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
        %49 = "ttir.unsqueeze"(%4) <{dim = 1 : si32}> : (tensor<1x512xi64>) -> tensor<1x1x512xi64>
        %50 = "ttir.typecast"(%49) <{conservative_folding = false}> : (tensor<1x1x512xi64>) -> tensor<1x1x512xf32>
        %51 = "ttir.matmul"(%48, %50) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x1xf32>, tensor<1x1x512xf32>) -> tensor<1x32x512xf32>
        %52 = "ttir.transpose"(%51) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512xf32>) -> tensor<1x512x32xf32>
        %53 = "ttir.concat"(%52, %52) <{dim = 2 : si32}> : (tensor<1x512x32xf32>, tensor<1x512x32xf32>) -> tensor<1x512x64xf32>
        %54 = "ttir.cos"(%53) : (tensor<1x512x64xf32>) -> tensor<1x512x64xf32>
        %55 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %56 = "ttir.multiply"(%54, %55) : (tensor<1x512x64xf32>, tensor<1xf32>) -> tensor<1x512x64xf32>
        %57 = "ttir.sin"(%53) : (tensor<1x512x64xf32>) -> tensor<1x512x64xf32>
        %58 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %59 = "ttir.multiply"(%57, %58) : (tensor<1x512x64xf32>, tensor<1xf32>) -> tensor<1x512x64xf32>
        %60 = "ttir.typecast"(%56) <{conservative_folding = false}> : (tensor<1x512x64xf32>) -> tensor<1x512x64xbf16>
        %61 = "ttir.typecast"(%59) <{conservative_folding = false}> : (tensor<1x512x64xf32>) -> tensor<1x512x64xbf16>
        %62 = "ttir.typecast"(%0) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %63 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %64 = "ttir.pow"(%62, %63) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %65 = "ttir.mean"(%64) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %66 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %67 = "ttir.add"(%65, %66) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %68 = "ttir.rsqrt"(%67) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %69 = "ttir.multiply"(%62, %68) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %70 = "ttir.typecast"(%69) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %71 = "ttir.multiply"(%arg8, %70) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %72 = "ttir.linear"(%71, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %73 = "ttir.reshape"(%72) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %74 = "ttir.transpose"(%73) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %75 = "ttir.linear"(%71, %arg2) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %76 = "ttir.reshape"(%75) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %77 = "ttir.transpose"(%76) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %78 = "ttir.linear"(%71, %arg3) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %79 = "ttir.reshape"(%78) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %80 = "ttir.transpose"(%79) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %81 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %82 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %83 = "ttir.multiply"(%74, %81) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %84 = "ttir.slice_static"(%74) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %85 = "ttir.slice_static"(%74) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %86 = "ttir.neg"(%85) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %87 = "ttir.concat"(%86, %84) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %88 = "ttir.multiply"(%87, %82) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %89 = "ttir.add"(%83, %88) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %90 = "ttir.multiply"(%77, %81) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %91 = "ttir.slice_static"(%77) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %92 = "ttir.slice_static"(%77) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %93 = "ttir.neg"(%92) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %94 = "ttir.concat"(%93, %91) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %95 = "ttir.multiply"(%94, %82) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %96 = "ttir.add"(%90, %95) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %97 = "ttir.unsqueeze"(%96) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %98 = "ttir.broadcast"(%97) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %99 = "ttir.reshape"(%98) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %100 = "ttir.unsqueeze"(%80) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %101 = "ttir.broadcast"(%100) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %102 = "ttir.reshape"(%101) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %103 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %104 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %105 = "ttir.where"(%45, %104, %103) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %106 = "ttir.typecast"(%89) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %107 = "ttir.typecast"(%99) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %108 = "ttir.typecast"(%102) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %109 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %110 = "ttir.multiply"(%106, %109) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %111 = "ttir.transpose"(%107) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %112 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %113 = "ttir.multiply"(%111, %112) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %114 = "ttir.matmul"(%110, %113) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %115 = "ttir.typecast"(%105) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %116 = "ttir.add"(%114, %115) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %117 = "ttir.softmax"(%116) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %118 = "ttir.matmul"(%117, %108) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %119 = "ttir.typecast"(%118) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %120 = "ttir.transpose"(%119) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %121 = "ttir.reshape"(%120) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %122 = "ttir.linear"(%121, %arg4) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %123 = "ttir.add"(%0, %122) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %124 = "ttir.typecast"(%123) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %125 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %126 = "ttir.pow"(%124, %125) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %127 = "ttir.mean"(%126) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %128 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %129 = "ttir.add"(%127, %128) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %130 = "ttir.rsqrt"(%129) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %131 = "ttir.multiply"(%124, %130) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %132 = "ttir.typecast"(%131) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %133 = "ttir.multiply"(%arg9, %132) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %134 = "ttir.linear"(%133, %arg5) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %135 = "ttir.silu"(%134) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %136 = "ttir.linear"(%133, %arg6) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %137 = "ttir.multiply"(%135, %136) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %138 = "ttir.linear"(%137, %arg7) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %139 = "ttir.add"(%123, %138) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %140 = "ttir.typecast"(%139) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %141 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %142 = "ttir.pow"(%140, %141) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %143 = "ttir.mean"(%142) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %144 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %145 = "ttir.add"(%143, %144) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %146 = "ttir.rsqrt"(%145) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %147 = "ttir.multiply"(%140, %146) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %148 = "ttir.typecast"(%147) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %149 = "ttir.multiply"(%arg17, %148) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %150 = "ttir.linear"(%149, %arg10) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %151 = "ttir.reshape"(%150) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %152 = "ttir.transpose"(%151) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %153 = "ttir.linear"(%149, %arg11) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %154 = "ttir.reshape"(%153) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %155 = "ttir.transpose"(%154) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %156 = "ttir.linear"(%149, %arg12) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %157 = "ttir.reshape"(%156) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %158 = "ttir.transpose"(%157) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %159 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %160 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %161 = "ttir.multiply"(%152, %159) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %162 = "ttir.slice_static"(%152) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %163 = "ttir.slice_static"(%152) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %164 = "ttir.neg"(%163) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %165 = "ttir.concat"(%164, %162) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %166 = "ttir.multiply"(%165, %160) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %167 = "ttir.add"(%161, %166) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %168 = "ttir.multiply"(%155, %159) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %169 = "ttir.slice_static"(%155) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %170 = "ttir.slice_static"(%155) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %171 = "ttir.neg"(%170) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %172 = "ttir.concat"(%171, %169) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %173 = "ttir.multiply"(%172, %160) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %174 = "ttir.add"(%168, %173) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %175 = "ttir.unsqueeze"(%174) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %176 = "ttir.broadcast"(%175) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %177 = "ttir.reshape"(%176) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %178 = "ttir.unsqueeze"(%158) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %179 = "ttir.broadcast"(%178) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %180 = "ttir.reshape"(%179) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %181 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %182 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %183 = "ttir.where"(%45, %182, %181) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %184 = "ttir.typecast"(%167) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %185 = "ttir.typecast"(%177) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %186 = "ttir.typecast"(%180) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %187 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %188 = "ttir.multiply"(%184, %187) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %189 = "ttir.transpose"(%185) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %190 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %191 = "ttir.multiply"(%189, %190) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %192 = "ttir.matmul"(%188, %191) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %193 = "ttir.typecast"(%183) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %194 = "ttir.add"(%192, %193) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %195 = "ttir.softmax"(%194) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %196 = "ttir.matmul"(%195, %186) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %197 = "ttir.typecast"(%196) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %198 = "ttir.transpose"(%197) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %199 = "ttir.reshape"(%198) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %200 = "ttir.linear"(%199, %arg13) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %201 = "ttir.add"(%139, %200) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %202 = "ttir.typecast"(%201) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %203 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %204 = "ttir.pow"(%202, %203) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %205 = "ttir.mean"(%204) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %206 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %207 = "ttir.add"(%205, %206) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %208 = "ttir.rsqrt"(%207) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %209 = "ttir.multiply"(%202, %208) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %210 = "ttir.typecast"(%209) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %211 = "ttir.multiply"(%arg18, %210) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %212 = "ttir.linear"(%211, %arg14) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %213 = "ttir.silu"(%212) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %214 = "ttir.linear"(%211, %arg15) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %215 = "ttir.multiply"(%213, %214) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %216 = "ttir.linear"(%215, %arg16) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %217 = "ttir.add"(%201, %216) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %218 = "ttir.typecast"(%217) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %219 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %220 = "ttir.pow"(%218, %219) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %221 = "ttir.mean"(%220) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %222 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %223 = "ttir.add"(%221, %222) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %224 = "ttir.rsqrt"(%223) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %225 = "ttir.multiply"(%218, %224) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %226 = "ttir.typecast"(%225) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %227 = "ttir.multiply"(%arg26, %226) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %228 = "ttir.linear"(%227, %arg19) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %229 = "ttir.reshape"(%228) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %230 = "ttir.transpose"(%229) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %231 = "ttir.linear"(%227, %arg20) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %232 = "ttir.reshape"(%231) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %233 = "ttir.transpose"(%232) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %234 = "ttir.linear"(%227, %arg21) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %235 = "ttir.reshape"(%234) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %236 = "ttir.transpose"(%235) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %237 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %238 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %239 = "ttir.multiply"(%230, %237) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %240 = "ttir.slice_static"(%230) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %241 = "ttir.slice_static"(%230) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %242 = "ttir.neg"(%241) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %243 = "ttir.concat"(%242, %240) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %244 = "ttir.multiply"(%243, %238) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %245 = "ttir.add"(%239, %244) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %246 = "ttir.multiply"(%233, %237) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %247 = "ttir.slice_static"(%233) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %248 = "ttir.slice_static"(%233) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %249 = "ttir.neg"(%248) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %250 = "ttir.concat"(%249, %247) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %251 = "ttir.multiply"(%250, %238) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %252 = "ttir.add"(%246, %251) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %253 = "ttir.unsqueeze"(%252) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %254 = "ttir.broadcast"(%253) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %255 = "ttir.reshape"(%254) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %256 = "ttir.unsqueeze"(%236) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %257 = "ttir.broadcast"(%256) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %258 = "ttir.reshape"(%257) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %259 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %260 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %261 = "ttir.where"(%45, %260, %259) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %262 = "ttir.typecast"(%245) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %263 = "ttir.typecast"(%255) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %264 = "ttir.typecast"(%258) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %265 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %266 = "ttir.multiply"(%262, %265) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %267 = "ttir.transpose"(%263) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %268 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %269 = "ttir.multiply"(%267, %268) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %270 = "ttir.matmul"(%266, %269) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %271 = "ttir.typecast"(%261) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %272 = "ttir.add"(%270, %271) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %273 = "ttir.softmax"(%272) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %274 = "ttir.matmul"(%273, %264) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %275 = "ttir.typecast"(%274) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %276 = "ttir.transpose"(%275) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %277 = "ttir.reshape"(%276) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %278 = "ttir.linear"(%277, %arg22) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %279 = "ttir.add"(%217, %278) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %280 = "ttir.typecast"(%279) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %281 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %282 = "ttir.pow"(%280, %281) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %283 = "ttir.mean"(%282) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %284 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %285 = "ttir.add"(%283, %284) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %286 = "ttir.rsqrt"(%285) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %287 = "ttir.multiply"(%280, %286) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %288 = "ttir.typecast"(%287) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %289 = "ttir.multiply"(%arg27, %288) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %290 = "ttir.linear"(%289, %arg23) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %291 = "ttir.silu"(%290) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %292 = "ttir.linear"(%289, %arg24) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %293 = "ttir.multiply"(%291, %292) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %294 = "ttir.linear"(%293, %arg25) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %295 = "ttir.add"(%279, %294) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %296 = "ttir.typecast"(%295) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %297 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %298 = "ttir.pow"(%296, %297) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %299 = "ttir.mean"(%298) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %300 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %301 = "ttir.add"(%299, %300) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %302 = "ttir.rsqrt"(%301) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %303 = "ttir.multiply"(%296, %302) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %304 = "ttir.typecast"(%303) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %305 = "ttir.multiply"(%arg35, %304) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %306 = "ttir.linear"(%305, %arg28) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %307 = "ttir.reshape"(%306) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %308 = "ttir.transpose"(%307) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %309 = "ttir.linear"(%305, %arg29) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %310 = "ttir.reshape"(%309) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %311 = "ttir.transpose"(%310) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %312 = "ttir.linear"(%305, %arg30) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %313 = "ttir.reshape"(%312) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %314 = "ttir.transpose"(%313) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %315 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %316 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %317 = "ttir.multiply"(%308, %315) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %318 = "ttir.slice_static"(%308) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %319 = "ttir.slice_static"(%308) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %320 = "ttir.neg"(%319) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %321 = "ttir.concat"(%320, %318) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %322 = "ttir.multiply"(%321, %316) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %323 = "ttir.add"(%317, %322) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %324 = "ttir.multiply"(%311, %315) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %325 = "ttir.slice_static"(%311) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %326 = "ttir.slice_static"(%311) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %327 = "ttir.neg"(%326) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %328 = "ttir.concat"(%327, %325) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %329 = "ttir.multiply"(%328, %316) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %330 = "ttir.add"(%324, %329) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %331 = "ttir.unsqueeze"(%330) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %332 = "ttir.broadcast"(%331) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %333 = "ttir.reshape"(%332) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %334 = "ttir.unsqueeze"(%314) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %335 = "ttir.broadcast"(%334) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %336 = "ttir.reshape"(%335) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %337 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %338 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %339 = "ttir.where"(%45, %338, %337) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %340 = "ttir.typecast"(%323) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %341 = "ttir.typecast"(%333) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %342 = "ttir.typecast"(%336) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %343 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %344 = "ttir.multiply"(%340, %343) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %345 = "ttir.transpose"(%341) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %346 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %347 = "ttir.multiply"(%345, %346) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %348 = "ttir.matmul"(%344, %347) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %349 = "ttir.typecast"(%339) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %350 = "ttir.add"(%348, %349) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %351 = "ttir.softmax"(%350) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %352 = "ttir.matmul"(%351, %342) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %353 = "ttir.typecast"(%352) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %354 = "ttir.transpose"(%353) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %355 = "ttir.reshape"(%354) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %356 = "ttir.linear"(%355, %arg31) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %357 = "ttir.add"(%295, %356) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %358 = "ttir.typecast"(%357) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %359 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %360 = "ttir.pow"(%358, %359) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %361 = "ttir.mean"(%360) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %362 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %363 = "ttir.add"(%361, %362) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %364 = "ttir.rsqrt"(%363) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %365 = "ttir.multiply"(%358, %364) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %366 = "ttir.typecast"(%365) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %367 = "ttir.multiply"(%arg36, %366) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %368 = "ttir.linear"(%367, %arg32) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %369 = "ttir.silu"(%368) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %370 = "ttir.linear"(%367, %arg33) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %371 = "ttir.multiply"(%369, %370) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %372 = "ttir.linear"(%371, %arg34) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %373 = "ttir.add"(%357, %372) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %374 = "ttir.typecast"(%373) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %375 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %376 = "ttir.pow"(%374, %375) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %377 = "ttir.mean"(%376) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %378 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %379 = "ttir.add"(%377, %378) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %380 = "ttir.rsqrt"(%379) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %381 = "ttir.multiply"(%374, %380) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %382 = "ttir.typecast"(%381) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %383 = "ttir.multiply"(%arg44, %382) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %384 = "ttir.linear"(%383, %arg37) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %385 = "ttir.reshape"(%384) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %386 = "ttir.transpose"(%385) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %387 = "ttir.linear"(%383, %arg38) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %388 = "ttir.reshape"(%387) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %389 = "ttir.transpose"(%388) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %390 = "ttir.linear"(%383, %arg39) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %391 = "ttir.reshape"(%390) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %392 = "ttir.transpose"(%391) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %393 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %394 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %395 = "ttir.multiply"(%386, %393) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %396 = "ttir.slice_static"(%386) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %397 = "ttir.slice_static"(%386) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %398 = "ttir.neg"(%397) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %399 = "ttir.concat"(%398, %396) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %400 = "ttir.multiply"(%399, %394) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %401 = "ttir.add"(%395, %400) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %402 = "ttir.multiply"(%389, %393) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %403 = "ttir.slice_static"(%389) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %404 = "ttir.slice_static"(%389) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %405 = "ttir.neg"(%404) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %406 = "ttir.concat"(%405, %403) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %407 = "ttir.multiply"(%406, %394) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %408 = "ttir.add"(%402, %407) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %409 = "ttir.unsqueeze"(%408) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %410 = "ttir.broadcast"(%409) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %411 = "ttir.reshape"(%410) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %412 = "ttir.unsqueeze"(%392) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %413 = "ttir.broadcast"(%412) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %414 = "ttir.reshape"(%413) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %415 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %416 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %417 = "ttir.where"(%45, %416, %415) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %418 = "ttir.typecast"(%401) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %419 = "ttir.typecast"(%411) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %420 = "ttir.typecast"(%414) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %421 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %422 = "ttir.multiply"(%418, %421) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %423 = "ttir.transpose"(%419) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %424 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %425 = "ttir.multiply"(%423, %424) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %426 = "ttir.matmul"(%422, %425) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %427 = "ttir.typecast"(%417) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %428 = "ttir.add"(%426, %427) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %429 = "ttir.softmax"(%428) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %430 = "ttir.matmul"(%429, %420) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %431 = "ttir.typecast"(%430) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %432 = "ttir.transpose"(%431) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %433 = "ttir.reshape"(%432) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %434 = "ttir.linear"(%433, %arg40) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %435 = "ttir.add"(%373, %434) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %436 = "ttir.typecast"(%435) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %437 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %438 = "ttir.pow"(%436, %437) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %439 = "ttir.mean"(%438) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %440 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %441 = "ttir.add"(%439, %440) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %442 = "ttir.rsqrt"(%441) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %443 = "ttir.multiply"(%436, %442) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %444 = "ttir.typecast"(%443) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %445 = "ttir.multiply"(%arg45, %444) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %446 = "ttir.linear"(%445, %arg41) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %447 = "ttir.silu"(%446) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %448 = "ttir.linear"(%445, %arg42) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %449 = "ttir.multiply"(%447, %448) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %450 = "ttir.linear"(%449, %arg43) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %451 = "ttir.add"(%435, %450) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %452 = "ttir.typecast"(%451) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %453 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %454 = "ttir.pow"(%452, %453) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %455 = "ttir.mean"(%454) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %456 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %457 = "ttir.add"(%455, %456) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %458 = "ttir.rsqrt"(%457) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %459 = "ttir.multiply"(%452, %458) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %460 = "ttir.typecast"(%459) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %461 = "ttir.multiply"(%arg53, %460) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %462 = "ttir.linear"(%461, %arg46) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %463 = "ttir.reshape"(%462) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %464 = "ttir.transpose"(%463) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %465 = "ttir.linear"(%461, %arg47) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %466 = "ttir.reshape"(%465) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %467 = "ttir.transpose"(%466) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %468 = "ttir.linear"(%461, %arg48) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %469 = "ttir.reshape"(%468) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %470 = "ttir.transpose"(%469) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %471 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %472 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %473 = "ttir.multiply"(%464, %471) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %474 = "ttir.slice_static"(%464) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %475 = "ttir.slice_static"(%464) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %476 = "ttir.neg"(%475) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %477 = "ttir.concat"(%476, %474) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %478 = "ttir.multiply"(%477, %472) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %479 = "ttir.add"(%473, %478) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %480 = "ttir.multiply"(%467, %471) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %481 = "ttir.slice_static"(%467) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %482 = "ttir.slice_static"(%467) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %483 = "ttir.neg"(%482) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %484 = "ttir.concat"(%483, %481) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %485 = "ttir.multiply"(%484, %472) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %486 = "ttir.add"(%480, %485) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %487 = "ttir.unsqueeze"(%486) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %488 = "ttir.broadcast"(%487) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %489 = "ttir.reshape"(%488) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %490 = "ttir.unsqueeze"(%470) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %491 = "ttir.broadcast"(%490) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %492 = "ttir.reshape"(%491) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %493 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %494 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %495 = "ttir.where"(%45, %494, %493) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %496 = "ttir.typecast"(%479) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %497 = "ttir.typecast"(%489) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %498 = "ttir.typecast"(%492) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %499 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %500 = "ttir.multiply"(%496, %499) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %501 = "ttir.transpose"(%497) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %502 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %503 = "ttir.multiply"(%501, %502) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %504 = "ttir.matmul"(%500, %503) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %505 = "ttir.typecast"(%495) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %506 = "ttir.add"(%504, %505) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %507 = "ttir.softmax"(%506) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %508 = "ttir.matmul"(%507, %498) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %509 = "ttir.typecast"(%508) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %510 = "ttir.transpose"(%509) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %511 = "ttir.reshape"(%510) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %512 = "ttir.linear"(%511, %arg49) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %513 = "ttir.add"(%451, %512) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %514 = "ttir.typecast"(%513) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %515 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %516 = "ttir.pow"(%514, %515) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %517 = "ttir.mean"(%516) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %518 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %519 = "ttir.add"(%517, %518) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %520 = "ttir.rsqrt"(%519) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %521 = "ttir.multiply"(%514, %520) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %522 = "ttir.typecast"(%521) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %523 = "ttir.multiply"(%arg54, %522) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %524 = "ttir.linear"(%523, %arg50) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %525 = "ttir.silu"(%524) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %526 = "ttir.linear"(%523, %arg51) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %527 = "ttir.multiply"(%525, %526) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %528 = "ttir.linear"(%527, %arg52) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %529 = "ttir.add"(%513, %528) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %530 = "ttir.typecast"(%529) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %531 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %532 = "ttir.pow"(%530, %531) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %533 = "ttir.mean"(%532) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %534 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %535 = "ttir.add"(%533, %534) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %536 = "ttir.rsqrt"(%535) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %537 = "ttir.multiply"(%530, %536) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %538 = "ttir.typecast"(%537) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %539 = "ttir.multiply"(%arg62, %538) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %540 = "ttir.linear"(%539, %arg55) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %541 = "ttir.reshape"(%540) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %542 = "ttir.transpose"(%541) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %543 = "ttir.linear"(%539, %arg56) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %544 = "ttir.reshape"(%543) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %545 = "ttir.transpose"(%544) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %546 = "ttir.linear"(%539, %arg57) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %547 = "ttir.reshape"(%546) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %548 = "ttir.transpose"(%547) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %549 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %550 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %551 = "ttir.multiply"(%542, %549) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %552 = "ttir.slice_static"(%542) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %553 = "ttir.slice_static"(%542) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %554 = "ttir.neg"(%553) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %555 = "ttir.concat"(%554, %552) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %556 = "ttir.multiply"(%555, %550) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %557 = "ttir.add"(%551, %556) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %558 = "ttir.multiply"(%545, %549) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %559 = "ttir.slice_static"(%545) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %560 = "ttir.slice_static"(%545) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %561 = "ttir.neg"(%560) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %562 = "ttir.concat"(%561, %559) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %563 = "ttir.multiply"(%562, %550) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %564 = "ttir.add"(%558, %563) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %565 = "ttir.unsqueeze"(%564) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %566 = "ttir.broadcast"(%565) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %567 = "ttir.reshape"(%566) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %568 = "ttir.unsqueeze"(%548) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %569 = "ttir.broadcast"(%568) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %570 = "ttir.reshape"(%569) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %571 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %572 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %573 = "ttir.where"(%45, %572, %571) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %574 = "ttir.typecast"(%557) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %575 = "ttir.typecast"(%567) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %576 = "ttir.typecast"(%570) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %577 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %578 = "ttir.multiply"(%574, %577) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %579 = "ttir.transpose"(%575) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %580 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %581 = "ttir.multiply"(%579, %580) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %582 = "ttir.matmul"(%578, %581) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %583 = "ttir.typecast"(%573) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %584 = "ttir.add"(%582, %583) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %585 = "ttir.softmax"(%584) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %586 = "ttir.matmul"(%585, %576) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %587 = "ttir.typecast"(%586) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %588 = "ttir.transpose"(%587) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %589 = "ttir.reshape"(%588) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %590 = "ttir.linear"(%589, %arg58) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %591 = "ttir.add"(%529, %590) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %592 = "ttir.typecast"(%591) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %593 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %594 = "ttir.pow"(%592, %593) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %595 = "ttir.mean"(%594) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %596 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %597 = "ttir.add"(%595, %596) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %598 = "ttir.rsqrt"(%597) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %599 = "ttir.multiply"(%592, %598) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %600 = "ttir.typecast"(%599) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %601 = "ttir.multiply"(%arg63, %600) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %602 = "ttir.linear"(%601, %arg59) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %603 = "ttir.silu"(%602) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %604 = "ttir.linear"(%601, %arg60) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %605 = "ttir.multiply"(%603, %604) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %606 = "ttir.linear"(%605, %arg61) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %607 = "ttir.add"(%591, %606) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %608 = "ttir.typecast"(%607) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %609 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %610 = "ttir.pow"(%608, %609) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %611 = "ttir.mean"(%610) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %612 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %613 = "ttir.add"(%611, %612) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %614 = "ttir.rsqrt"(%613) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %615 = "ttir.multiply"(%608, %614) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %616 = "ttir.typecast"(%615) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %617 = "ttir.multiply"(%arg71, %616) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %618 = "ttir.linear"(%617, %arg64) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %619 = "ttir.reshape"(%618) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %620 = "ttir.transpose"(%619) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %621 = "ttir.linear"(%617, %arg65) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %622 = "ttir.reshape"(%621) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %623 = "ttir.transpose"(%622) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %624 = "ttir.linear"(%617, %arg66) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %625 = "ttir.reshape"(%624) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %626 = "ttir.transpose"(%625) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %627 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %628 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %629 = "ttir.multiply"(%620, %627) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %630 = "ttir.slice_static"(%620) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %631 = "ttir.slice_static"(%620) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %632 = "ttir.neg"(%631) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %633 = "ttir.concat"(%632, %630) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %634 = "ttir.multiply"(%633, %628) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %635 = "ttir.add"(%629, %634) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %636 = "ttir.multiply"(%623, %627) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %637 = "ttir.slice_static"(%623) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %638 = "ttir.slice_static"(%623) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %639 = "ttir.neg"(%638) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %640 = "ttir.concat"(%639, %637) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %641 = "ttir.multiply"(%640, %628) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %642 = "ttir.add"(%636, %641) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %643 = "ttir.unsqueeze"(%642) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %644 = "ttir.broadcast"(%643) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %645 = "ttir.reshape"(%644) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %646 = "ttir.unsqueeze"(%626) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %647 = "ttir.broadcast"(%646) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %648 = "ttir.reshape"(%647) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %649 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %650 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %651 = "ttir.where"(%45, %650, %649) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %652 = "ttir.typecast"(%635) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %653 = "ttir.typecast"(%645) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %654 = "ttir.typecast"(%648) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %655 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %656 = "ttir.multiply"(%652, %655) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %657 = "ttir.transpose"(%653) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %658 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %659 = "ttir.multiply"(%657, %658) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %660 = "ttir.matmul"(%656, %659) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %661 = "ttir.typecast"(%651) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %662 = "ttir.add"(%660, %661) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %663 = "ttir.softmax"(%662) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %664 = "ttir.matmul"(%663, %654) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %665 = "ttir.typecast"(%664) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %666 = "ttir.transpose"(%665) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %667 = "ttir.reshape"(%666) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %668 = "ttir.linear"(%667, %arg67) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %669 = "ttir.add"(%607, %668) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %670 = "ttir.typecast"(%669) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %671 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %672 = "ttir.pow"(%670, %671) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %673 = "ttir.mean"(%672) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %674 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %675 = "ttir.add"(%673, %674) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %676 = "ttir.rsqrt"(%675) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %677 = "ttir.multiply"(%670, %676) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %678 = "ttir.typecast"(%677) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %679 = "ttir.multiply"(%arg72, %678) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %680 = "ttir.linear"(%679, %arg68) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %681 = "ttir.silu"(%680) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %682 = "ttir.linear"(%679, %arg69) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %683 = "ttir.multiply"(%681, %682) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %684 = "ttir.linear"(%683, %arg70) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %685 = "ttir.add"(%669, %684) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %686 = "ttir.typecast"(%685) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %687 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %688 = "ttir.pow"(%686, %687) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %689 = "ttir.mean"(%688) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %690 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %691 = "ttir.add"(%689, %690) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %692 = "ttir.rsqrt"(%691) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %693 = "ttir.multiply"(%686, %692) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %694 = "ttir.typecast"(%693) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %695 = "ttir.multiply"(%arg80, %694) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %696 = "ttir.linear"(%695, %arg73) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %697 = "ttir.reshape"(%696) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %698 = "ttir.transpose"(%697) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %699 = "ttir.linear"(%695, %arg74) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %700 = "ttir.reshape"(%699) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %701 = "ttir.transpose"(%700) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %702 = "ttir.linear"(%695, %arg75) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %703 = "ttir.reshape"(%702) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %704 = "ttir.transpose"(%703) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %705 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %706 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %707 = "ttir.multiply"(%698, %705) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %708 = "ttir.slice_static"(%698) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %709 = "ttir.slice_static"(%698) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %710 = "ttir.neg"(%709) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %711 = "ttir.concat"(%710, %708) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %712 = "ttir.multiply"(%711, %706) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %713 = "ttir.add"(%707, %712) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %714 = "ttir.multiply"(%701, %705) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %715 = "ttir.slice_static"(%701) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %716 = "ttir.slice_static"(%701) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %717 = "ttir.neg"(%716) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %718 = "ttir.concat"(%717, %715) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %719 = "ttir.multiply"(%718, %706) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %720 = "ttir.add"(%714, %719) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %721 = "ttir.unsqueeze"(%720) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %722 = "ttir.broadcast"(%721) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %723 = "ttir.reshape"(%722) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %724 = "ttir.unsqueeze"(%704) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %725 = "ttir.broadcast"(%724) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %726 = "ttir.reshape"(%725) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %727 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %728 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %729 = "ttir.where"(%45, %728, %727) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %730 = "ttir.typecast"(%713) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %731 = "ttir.typecast"(%723) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %732 = "ttir.typecast"(%726) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %733 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %734 = "ttir.multiply"(%730, %733) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %735 = "ttir.transpose"(%731) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %736 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %737 = "ttir.multiply"(%735, %736) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %738 = "ttir.matmul"(%734, %737) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %739 = "ttir.typecast"(%729) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %740 = "ttir.add"(%738, %739) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %741 = "ttir.softmax"(%740) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %742 = "ttir.matmul"(%741, %732) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %743 = "ttir.typecast"(%742) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %744 = "ttir.transpose"(%743) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %745 = "ttir.reshape"(%744) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %746 = "ttir.linear"(%745, %arg76) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %747 = "ttir.add"(%685, %746) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %748 = "ttir.typecast"(%747) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %749 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %750 = "ttir.pow"(%748, %749) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %751 = "ttir.mean"(%750) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %752 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %753 = "ttir.add"(%751, %752) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %754 = "ttir.rsqrt"(%753) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %755 = "ttir.multiply"(%748, %754) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %756 = "ttir.typecast"(%755) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %757 = "ttir.multiply"(%arg81, %756) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %758 = "ttir.linear"(%757, %arg77) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %759 = "ttir.silu"(%758) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %760 = "ttir.linear"(%757, %arg78) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %761 = "ttir.multiply"(%759, %760) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %762 = "ttir.linear"(%761, %arg79) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %763 = "ttir.add"(%747, %762) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %764 = "ttir.typecast"(%763) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %765 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %766 = "ttir.pow"(%764, %765) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %767 = "ttir.mean"(%766) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %768 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %769 = "ttir.add"(%767, %768) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %770 = "ttir.rsqrt"(%769) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %771 = "ttir.multiply"(%764, %770) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %772 = "ttir.typecast"(%771) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %773 = "ttir.multiply"(%arg89, %772) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %774 = "ttir.linear"(%773, %arg82) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %775 = "ttir.reshape"(%774) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %776 = "ttir.transpose"(%775) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %777 = "ttir.linear"(%773, %arg83) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %778 = "ttir.reshape"(%777) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %779 = "ttir.transpose"(%778) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %780 = "ttir.linear"(%773, %arg84) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %781 = "ttir.reshape"(%780) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %782 = "ttir.transpose"(%781) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %783 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %784 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %785 = "ttir.multiply"(%776, %783) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %786 = "ttir.slice_static"(%776) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %787 = "ttir.slice_static"(%776) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %788 = "ttir.neg"(%787) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %789 = "ttir.concat"(%788, %786) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %790 = "ttir.multiply"(%789, %784) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %791 = "ttir.add"(%785, %790) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %792 = "ttir.multiply"(%779, %783) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %793 = "ttir.slice_static"(%779) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %794 = "ttir.slice_static"(%779) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %795 = "ttir.neg"(%794) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %796 = "ttir.concat"(%795, %793) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %797 = "ttir.multiply"(%796, %784) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %798 = "ttir.add"(%792, %797) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %799 = "ttir.unsqueeze"(%798) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %800 = "ttir.broadcast"(%799) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %801 = "ttir.reshape"(%800) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %802 = "ttir.unsqueeze"(%782) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %803 = "ttir.broadcast"(%802) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %804 = "ttir.reshape"(%803) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %805 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %806 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %807 = "ttir.where"(%45, %806, %805) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %808 = "ttir.typecast"(%791) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %809 = "ttir.typecast"(%801) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %810 = "ttir.typecast"(%804) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %811 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %812 = "ttir.multiply"(%808, %811) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %813 = "ttir.transpose"(%809) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %814 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %815 = "ttir.multiply"(%813, %814) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %816 = "ttir.matmul"(%812, %815) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %817 = "ttir.typecast"(%807) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %818 = "ttir.add"(%816, %817) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %819 = "ttir.softmax"(%818) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %820 = "ttir.matmul"(%819, %810) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %821 = "ttir.typecast"(%820) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %822 = "ttir.transpose"(%821) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %823 = "ttir.reshape"(%822) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %824 = "ttir.linear"(%823, %arg85) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %825 = "ttir.add"(%763, %824) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %826 = "ttir.typecast"(%825) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %827 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %828 = "ttir.pow"(%826, %827) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %829 = "ttir.mean"(%828) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %830 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %831 = "ttir.add"(%829, %830) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %832 = "ttir.rsqrt"(%831) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %833 = "ttir.multiply"(%826, %832) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %834 = "ttir.typecast"(%833) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %835 = "ttir.multiply"(%arg90, %834) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %836 = "ttir.linear"(%835, %arg86) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %837 = "ttir.silu"(%836) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %838 = "ttir.linear"(%835, %arg87) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %839 = "ttir.multiply"(%837, %838) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %840 = "ttir.linear"(%839, %arg88) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %841 = "ttir.add"(%825, %840) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %842 = "ttir.typecast"(%841) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %843 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %844 = "ttir.pow"(%842, %843) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %845 = "ttir.mean"(%844) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %846 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %847 = "ttir.add"(%845, %846) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %848 = "ttir.rsqrt"(%847) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %849 = "ttir.multiply"(%842, %848) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %850 = "ttir.typecast"(%849) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %851 = "ttir.multiply"(%arg98, %850) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %852 = "ttir.linear"(%851, %arg91) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %853 = "ttir.reshape"(%852) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %854 = "ttir.transpose"(%853) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %855 = "ttir.linear"(%851, %arg92) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %856 = "ttir.reshape"(%855) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %857 = "ttir.transpose"(%856) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %858 = "ttir.linear"(%851, %arg93) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %859 = "ttir.reshape"(%858) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %860 = "ttir.transpose"(%859) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %861 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %862 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %863 = "ttir.multiply"(%854, %861) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %864 = "ttir.slice_static"(%854) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %865 = "ttir.slice_static"(%854) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %866 = "ttir.neg"(%865) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %867 = "ttir.concat"(%866, %864) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %868 = "ttir.multiply"(%867, %862) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %869 = "ttir.add"(%863, %868) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %870 = "ttir.multiply"(%857, %861) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %871 = "ttir.slice_static"(%857) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %872 = "ttir.slice_static"(%857) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %873 = "ttir.neg"(%872) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %874 = "ttir.concat"(%873, %871) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %875 = "ttir.multiply"(%874, %862) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %876 = "ttir.add"(%870, %875) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %877 = "ttir.unsqueeze"(%876) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %878 = "ttir.broadcast"(%877) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %879 = "ttir.reshape"(%878) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %880 = "ttir.unsqueeze"(%860) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %881 = "ttir.broadcast"(%880) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %882 = "ttir.reshape"(%881) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %883 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %884 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %885 = "ttir.where"(%45, %884, %883) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %886 = "ttir.typecast"(%869) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %887 = "ttir.typecast"(%879) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %888 = "ttir.typecast"(%882) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %889 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %890 = "ttir.multiply"(%886, %889) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %891 = "ttir.transpose"(%887) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %892 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %893 = "ttir.multiply"(%891, %892) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %894 = "ttir.matmul"(%890, %893) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %895 = "ttir.typecast"(%885) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %896 = "ttir.add"(%894, %895) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %897 = "ttir.softmax"(%896) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %898 = "ttir.matmul"(%897, %888) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %899 = "ttir.typecast"(%898) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %900 = "ttir.transpose"(%899) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %901 = "ttir.reshape"(%900) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %902 = "ttir.linear"(%901, %arg94) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %903 = "ttir.add"(%841, %902) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %904 = "ttir.typecast"(%903) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %905 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %906 = "ttir.pow"(%904, %905) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %907 = "ttir.mean"(%906) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %908 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %909 = "ttir.add"(%907, %908) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %910 = "ttir.rsqrt"(%909) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %911 = "ttir.multiply"(%904, %910) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %912 = "ttir.typecast"(%911) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %913 = "ttir.multiply"(%arg99, %912) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %914 = "ttir.linear"(%913, %arg95) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %915 = "ttir.silu"(%914) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %916 = "ttir.linear"(%913, %arg96) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %917 = "ttir.multiply"(%915, %916) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %918 = "ttir.linear"(%917, %arg97) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %919 = "ttir.add"(%903, %918) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %920 = "ttir.typecast"(%919) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %921 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %922 = "ttir.pow"(%920, %921) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %923 = "ttir.mean"(%922) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %924 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %925 = "ttir.add"(%923, %924) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %926 = "ttir.rsqrt"(%925) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %927 = "ttir.multiply"(%920, %926) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %928 = "ttir.typecast"(%927) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %929 = "ttir.multiply"(%arg107, %928) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %930 = "ttir.linear"(%929, %arg100) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %931 = "ttir.reshape"(%930) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %932 = "ttir.transpose"(%931) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %933 = "ttir.linear"(%929, %arg101) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %934 = "ttir.reshape"(%933) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %935 = "ttir.transpose"(%934) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %936 = "ttir.linear"(%929, %arg102) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %937 = "ttir.reshape"(%936) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %938 = "ttir.transpose"(%937) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %939 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %940 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %941 = "ttir.multiply"(%932, %939) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %942 = "ttir.slice_static"(%932) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %943 = "ttir.slice_static"(%932) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %944 = "ttir.neg"(%943) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %945 = "ttir.concat"(%944, %942) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %946 = "ttir.multiply"(%945, %940) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %947 = "ttir.add"(%941, %946) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %948 = "ttir.multiply"(%935, %939) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %949 = "ttir.slice_static"(%935) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %950 = "ttir.slice_static"(%935) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %951 = "ttir.neg"(%950) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %952 = "ttir.concat"(%951, %949) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %953 = "ttir.multiply"(%952, %940) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %954 = "ttir.add"(%948, %953) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %955 = "ttir.unsqueeze"(%954) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %956 = "ttir.broadcast"(%955) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %957 = "ttir.reshape"(%956) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %958 = "ttir.unsqueeze"(%938) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %959 = "ttir.broadcast"(%958) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %960 = "ttir.reshape"(%959) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %961 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %962 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %963 = "ttir.where"(%45, %962, %961) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %964 = "ttir.typecast"(%947) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %965 = "ttir.typecast"(%957) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %966 = "ttir.typecast"(%960) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %967 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %968 = "ttir.multiply"(%964, %967) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %969 = "ttir.transpose"(%965) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %970 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %971 = "ttir.multiply"(%969, %970) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %972 = "ttir.matmul"(%968, %971) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %973 = "ttir.typecast"(%963) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %974 = "ttir.add"(%972, %973) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %975 = "ttir.softmax"(%974) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %976 = "ttir.matmul"(%975, %966) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %977 = "ttir.typecast"(%976) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %978 = "ttir.transpose"(%977) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %979 = "ttir.reshape"(%978) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %980 = "ttir.linear"(%979, %arg103) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %981 = "ttir.add"(%919, %980) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %982 = "ttir.typecast"(%981) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %983 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %984 = "ttir.pow"(%982, %983) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %985 = "ttir.mean"(%984) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %986 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %987 = "ttir.add"(%985, %986) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %988 = "ttir.rsqrt"(%987) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %989 = "ttir.multiply"(%982, %988) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %990 = "ttir.typecast"(%989) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %991 = "ttir.multiply"(%arg108, %990) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %992 = "ttir.linear"(%991, %arg104) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %993 = "ttir.silu"(%992) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %994 = "ttir.linear"(%991, %arg105) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %995 = "ttir.multiply"(%993, %994) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %996 = "ttir.linear"(%995, %arg106) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %997 = "ttir.add"(%981, %996) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %998 = "ttir.typecast"(%997) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %999 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1000 = "ttir.pow"(%998, %999) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %1001 = "ttir.mean"(%1000) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %1002 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1003 = "ttir.add"(%1001, %1002) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %1004 = "ttir.rsqrt"(%1003) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1005 = "ttir.multiply"(%998, %1004) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %1006 = "ttir.typecast"(%1005) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %1007 = "ttir.multiply"(%arg116, %1006) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1008 = "ttir.linear"(%1007, %arg109) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1009 = "ttir.reshape"(%1008) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %1010 = "ttir.transpose"(%1009) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1011 = "ttir.linear"(%1007, %arg110) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %1012 = "ttir.reshape"(%1011) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %1013 = "ttir.transpose"(%1012) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1014 = "ttir.linear"(%1007, %arg111) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %1015 = "ttir.reshape"(%1014) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %1016 = "ttir.transpose"(%1015) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1017 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %1018 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %1019 = "ttir.multiply"(%1010, %1017) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1020 = "ttir.slice_static"(%1010) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %1021 = "ttir.slice_static"(%1010) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %1022 = "ttir.neg"(%1021) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %1023 = "ttir.concat"(%1022, %1020) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %1024 = "ttir.multiply"(%1023, %1018) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1025 = "ttir.add"(%1019, %1024) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1026 = "ttir.multiply"(%1013, %1017) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1027 = "ttir.slice_static"(%1013) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %1028 = "ttir.slice_static"(%1013) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %1029 = "ttir.neg"(%1028) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %1030 = "ttir.concat"(%1029, %1027) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %1031 = "ttir.multiply"(%1030, %1018) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1032 = "ttir.add"(%1026, %1031) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1033 = "ttir.unsqueeze"(%1032) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %1034 = "ttir.broadcast"(%1033) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %1035 = "ttir.reshape"(%1034) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1036 = "ttir.unsqueeze"(%1016) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %1037 = "ttir.broadcast"(%1036) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %1038 = "ttir.reshape"(%1037) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1039 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %1040 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %1041 = "ttir.where"(%45, %1040, %1039) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %1042 = "ttir.typecast"(%1025) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1043 = "ttir.typecast"(%1035) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1044 = "ttir.typecast"(%1038) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1045 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1046 = "ttir.multiply"(%1042, %1045) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %1047 = "ttir.transpose"(%1043) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %1048 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1049 = "ttir.multiply"(%1047, %1048) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %1050 = "ttir.matmul"(%1046, %1049) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %1051 = "ttir.typecast"(%1041) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %1052 = "ttir.add"(%1050, %1051) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %1053 = "ttir.softmax"(%1052) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %1054 = "ttir.matmul"(%1053, %1044) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %1055 = "ttir.typecast"(%1054) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %1056 = "ttir.transpose"(%1055) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %1057 = "ttir.reshape"(%1056) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %1058 = "ttir.linear"(%1057, %arg112) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1059 = "ttir.add"(%997, %1058) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1060 = "ttir.typecast"(%1059) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %1061 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1062 = "ttir.pow"(%1060, %1061) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %1063 = "ttir.mean"(%1062) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %1064 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1065 = "ttir.add"(%1063, %1064) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %1066 = "ttir.rsqrt"(%1065) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1067 = "ttir.multiply"(%1060, %1066) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %1068 = "ttir.typecast"(%1067) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %1069 = "ttir.multiply"(%arg117, %1068) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1070 = "ttir.linear"(%1069, %arg113) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %1071 = "ttir.silu"(%1070) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %1072 = "ttir.linear"(%1069, %arg114) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %1073 = "ttir.multiply"(%1071, %1072) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %1074 = "ttir.linear"(%1073, %arg115) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %1075 = "ttir.add"(%1059, %1074) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1076 = "ttir.typecast"(%1075) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %1077 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1078 = "ttir.pow"(%1076, %1077) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %1079 = "ttir.mean"(%1078) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %1080 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1081 = "ttir.add"(%1079, %1080) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %1082 = "ttir.rsqrt"(%1081) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1083 = "ttir.multiply"(%1076, %1082) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %1084 = "ttir.typecast"(%1083) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %1085 = "ttir.multiply"(%arg125, %1084) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1086 = "ttir.linear"(%1085, %arg118) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1087 = "ttir.reshape"(%1086) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %1088 = "ttir.transpose"(%1087) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1089 = "ttir.linear"(%1085, %arg119) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %1090 = "ttir.reshape"(%1089) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %1091 = "ttir.transpose"(%1090) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1092 = "ttir.linear"(%1085, %arg120) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %1093 = "ttir.reshape"(%1092) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %1094 = "ttir.transpose"(%1093) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1095 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %1096 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %1097 = "ttir.multiply"(%1088, %1095) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1098 = "ttir.slice_static"(%1088) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %1099 = "ttir.slice_static"(%1088) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %1100 = "ttir.neg"(%1099) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %1101 = "ttir.concat"(%1100, %1098) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %1102 = "ttir.multiply"(%1101, %1096) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1103 = "ttir.add"(%1097, %1102) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1104 = "ttir.multiply"(%1091, %1095) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1105 = "ttir.slice_static"(%1091) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %1106 = "ttir.slice_static"(%1091) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %1107 = "ttir.neg"(%1106) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %1108 = "ttir.concat"(%1107, %1105) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %1109 = "ttir.multiply"(%1108, %1096) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1110 = "ttir.add"(%1104, %1109) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1111 = "ttir.unsqueeze"(%1110) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %1112 = "ttir.broadcast"(%1111) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %1113 = "ttir.reshape"(%1112) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1114 = "ttir.unsqueeze"(%1094) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %1115 = "ttir.broadcast"(%1114) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %1116 = "ttir.reshape"(%1115) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1117 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %1118 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %1119 = "ttir.where"(%45, %1118, %1117) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %1120 = "ttir.typecast"(%1103) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1121 = "ttir.typecast"(%1113) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1122 = "ttir.typecast"(%1116) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1123 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1124 = "ttir.multiply"(%1120, %1123) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %1125 = "ttir.transpose"(%1121) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %1126 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1127 = "ttir.multiply"(%1125, %1126) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %1128 = "ttir.matmul"(%1124, %1127) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %1129 = "ttir.typecast"(%1119) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %1130 = "ttir.add"(%1128, %1129) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %1131 = "ttir.softmax"(%1130) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %1132 = "ttir.matmul"(%1131, %1122) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %1133 = "ttir.typecast"(%1132) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %1134 = "ttir.transpose"(%1133) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %1135 = "ttir.reshape"(%1134) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %1136 = "ttir.linear"(%1135, %arg121) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1137 = "ttir.add"(%1075, %1136) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1138 = "ttir.typecast"(%1137) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %1139 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1140 = "ttir.pow"(%1138, %1139) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %1141 = "ttir.mean"(%1140) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %1142 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1143 = "ttir.add"(%1141, %1142) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %1144 = "ttir.rsqrt"(%1143) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1145 = "ttir.multiply"(%1138, %1144) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %1146 = "ttir.typecast"(%1145) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %1147 = "ttir.multiply"(%arg126, %1146) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1148 = "ttir.linear"(%1147, %arg122) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %1149 = "ttir.silu"(%1148) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %1150 = "ttir.linear"(%1147, %arg123) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %1151 = "ttir.multiply"(%1149, %1150) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %1152 = "ttir.linear"(%1151, %arg124) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %1153 = "ttir.add"(%1137, %1152) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1154 = "ttir.typecast"(%1153) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %1155 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1156 = "ttir.pow"(%1154, %1155) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %1157 = "ttir.mean"(%1156) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %1158 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1159 = "ttir.add"(%1157, %1158) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %1160 = "ttir.rsqrt"(%1159) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1161 = "ttir.multiply"(%1154, %1160) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %1162 = "ttir.typecast"(%1161) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %1163 = "ttir.multiply"(%arg134, %1162) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1164 = "ttir.linear"(%1163, %arg127) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1165 = "ttir.reshape"(%1164) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %1166 = "ttir.transpose"(%1165) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1167 = "ttir.linear"(%1163, %arg128) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %1168 = "ttir.reshape"(%1167) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %1169 = "ttir.transpose"(%1168) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1170 = "ttir.linear"(%1163, %arg129) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %1171 = "ttir.reshape"(%1170) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %1172 = "ttir.transpose"(%1171) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1173 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %1174 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %1175 = "ttir.multiply"(%1166, %1173) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1176 = "ttir.slice_static"(%1166) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %1177 = "ttir.slice_static"(%1166) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %1178 = "ttir.neg"(%1177) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %1179 = "ttir.concat"(%1178, %1176) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %1180 = "ttir.multiply"(%1179, %1174) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1181 = "ttir.add"(%1175, %1180) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1182 = "ttir.multiply"(%1169, %1173) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1183 = "ttir.slice_static"(%1169) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %1184 = "ttir.slice_static"(%1169) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %1185 = "ttir.neg"(%1184) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %1186 = "ttir.concat"(%1185, %1183) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %1187 = "ttir.multiply"(%1186, %1174) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1188 = "ttir.add"(%1182, %1187) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1189 = "ttir.unsqueeze"(%1188) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %1190 = "ttir.broadcast"(%1189) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %1191 = "ttir.reshape"(%1190) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1192 = "ttir.unsqueeze"(%1172) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %1193 = "ttir.broadcast"(%1192) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %1194 = "ttir.reshape"(%1193) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1195 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %1196 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %1197 = "ttir.where"(%45, %1196, %1195) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %1198 = "ttir.typecast"(%1181) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1199 = "ttir.typecast"(%1191) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1200 = "ttir.typecast"(%1194) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1201 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1202 = "ttir.multiply"(%1198, %1201) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %1203 = "ttir.transpose"(%1199) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %1204 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1205 = "ttir.multiply"(%1203, %1204) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %1206 = "ttir.matmul"(%1202, %1205) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %1207 = "ttir.typecast"(%1197) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %1208 = "ttir.add"(%1206, %1207) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %1209 = "ttir.softmax"(%1208) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %1210 = "ttir.matmul"(%1209, %1200) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %1211 = "ttir.typecast"(%1210) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %1212 = "ttir.transpose"(%1211) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %1213 = "ttir.reshape"(%1212) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %1214 = "ttir.linear"(%1213, %arg130) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1215 = "ttir.add"(%1153, %1214) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1216 = "ttir.typecast"(%1215) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %1217 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1218 = "ttir.pow"(%1216, %1217) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %1219 = "ttir.mean"(%1218) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %1220 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1221 = "ttir.add"(%1219, %1220) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %1222 = "ttir.rsqrt"(%1221) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1223 = "ttir.multiply"(%1216, %1222) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %1224 = "ttir.typecast"(%1223) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %1225 = "ttir.multiply"(%arg135, %1224) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1226 = "ttir.linear"(%1225, %arg131) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %1227 = "ttir.silu"(%1226) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %1228 = "ttir.linear"(%1225, %arg132) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %1229 = "ttir.multiply"(%1227, %1228) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %1230 = "ttir.linear"(%1229, %arg133) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %1231 = "ttir.add"(%1215, %1230) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1232 = "ttir.typecast"(%1231) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %1233 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1234 = "ttir.pow"(%1232, %1233) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %1235 = "ttir.mean"(%1234) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %1236 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1237 = "ttir.add"(%1235, %1236) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %1238 = "ttir.rsqrt"(%1237) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1239 = "ttir.multiply"(%1232, %1238) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %1240 = "ttir.typecast"(%1239) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %1241 = "ttir.multiply"(%arg143, %1240) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1242 = "ttir.linear"(%1241, %arg136) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1243 = "ttir.reshape"(%1242) <{shape = [1 : i32, 512 : i32, 32 : i32, 64 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x32x64xbf16>
        %1244 = "ttir.transpose"(%1243) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x32x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1245 = "ttir.linear"(%1241, %arg137) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %1246 = "ttir.reshape"(%1245) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %1247 = "ttir.transpose"(%1246) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1248 = "ttir.linear"(%1241, %arg138) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<512x2048xbf16>) -> tensor<1x512x512xbf16>
        %1249 = "ttir.reshape"(%1248) <{shape = [1 : i32, 512 : i32, 8 : i32, 64 : i32]}> : (tensor<1x512x512xbf16>) -> tensor<1x512x8x64xbf16>
        %1250 = "ttir.transpose"(%1249) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x512x8x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1251 = "ttir.unsqueeze"(%60) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %1252 = "ttir.unsqueeze"(%61) <{dim = 1 : si32}> : (tensor<1x512x64xbf16>) -> tensor<1x1x512x64xbf16>
        %1253 = "ttir.multiply"(%1244, %1251) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1254 = "ttir.slice_static"(%1244) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %1255 = "ttir.slice_static"(%1244) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x32xbf16>
        %1256 = "ttir.neg"(%1255) : (tensor<1x32x512x32xbf16>) -> tensor<1x32x512x32xbf16>
        %1257 = "ttir.concat"(%1256, %1254) <{dim = 3 : si32}> : (tensor<1x32x512x32xbf16>, tensor<1x32x512x32xbf16>) -> tensor<1x32x512x64xbf16>
        %1258 = "ttir.multiply"(%1257, %1252) : (tensor<1x32x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1259 = "ttir.add"(%1253, %1258) : (tensor<1x32x512x64xbf16>, tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1260 = "ttir.multiply"(%1247, %1251) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1261 = "ttir.slice_static"(%1247) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %1262 = "ttir.slice_static"(%1247) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 512 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x512x32xbf16>
        %1263 = "ttir.neg"(%1262) : (tensor<1x8x512x32xbf16>) -> tensor<1x8x512x32xbf16>
        %1264 = "ttir.concat"(%1263, %1261) <{dim = 3 : si32}> : (tensor<1x8x512x32xbf16>, tensor<1x8x512x32xbf16>) -> tensor<1x8x512x64xbf16>
        %1265 = "ttir.multiply"(%1264, %1252) : (tensor<1x8x512x64xbf16>, tensor<1x1x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1266 = "ttir.add"(%1260, %1265) : (tensor<1x8x512x64xbf16>, tensor<1x8x512x64xbf16>) -> tensor<1x8x512x64xbf16>
        %1267 = "ttir.unsqueeze"(%1266) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %1268 = "ttir.broadcast"(%1267) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %1269 = "ttir.reshape"(%1268) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1270 = "ttir.unsqueeze"(%1250) <{dim = 2 : si32}> : (tensor<1x8x512x64xbf16>) -> tensor<1x8x1x512x64xbf16>
        %1271 = "ttir.broadcast"(%1270) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<1x8x1x512x64xbf16>) -> tensor<1x8x4x512x64xbf16>
        %1272 = "ttir.reshape"(%1271) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xbf16>) -> tensor<1x32x512x64xbf16>
        %1273 = "ttir.constant"() <{value = dense<0xFF80> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %1274 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
        %1275 = "ttir.where"(%45, %1274, %1273) : (tensor<1x1x512x512xi1>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x512x512xbf16>
        %1276 = "ttir.typecast"(%1259) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1277 = "ttir.typecast"(%1269) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1278 = "ttir.typecast"(%1272) <{conservative_folding = false}> : (tensor<1x32x512x64xbf16>) -> tensor<1x32x512x64xf32>
        %1279 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1280 = "ttir.multiply"(%1276, %1279) : (tensor<1x32x512x64xf32>, tensor<1xf32>) -> tensor<1x32x512x64xf32>
        %1281 = "ttir.transpose"(%1277) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x64x512xf32>
        %1282 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1283 = "ttir.multiply"(%1281, %1282) : (tensor<1x32x64x512xf32>, tensor<1xf32>) -> tensor<1x32x64x512xf32>
        %1284 = "ttir.matmul"(%1280, %1283) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>) -> tensor<1x32x512x512xf32>
        %1285 = "ttir.typecast"(%1275) <{conservative_folding = false}> : (tensor<1x1x512x512xbf16>) -> tensor<1x1x512x512xf32>
        %1286 = "ttir.add"(%1284, %1285) : (tensor<1x32x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x32x512x512xf32>
        %1287 = "ttir.softmax"(%1286) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
        %1288 = "ttir.matmul"(%1287, %1278) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xf32>
        %1289 = "ttir.typecast"(%1288) <{conservative_folding = false}> : (tensor<1x32x512x64xf32>) -> tensor<1x32x512x64xbf16>
        %1290 = "ttir.transpose"(%1289) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<1x32x512x64xbf16>) -> tensor<1x512x32x64xbf16>
        %1291 = "ttir.reshape"(%1290) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<1x512x32x64xbf16>) -> tensor<1x512x2048xbf16>
        %1292 = "ttir.linear"(%1291, %arg139) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1293 = "ttir.add"(%1231, %1292) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1294 = "ttir.typecast"(%1293) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %1295 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1296 = "ttir.pow"(%1294, %1295) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %1297 = "ttir.mean"(%1296) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %1298 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1299 = "ttir.add"(%1297, %1298) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %1300 = "ttir.rsqrt"(%1299) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1301 = "ttir.multiply"(%1294, %1300) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %1302 = "ttir.typecast"(%1301) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %1303 = "ttir.multiply"(%arg144, %1302) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1304 = "ttir.linear"(%1303, %arg140) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %1305 = "ttir.silu"(%1304) : (tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %1306 = "ttir.linear"(%1303, %arg141) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<8192x2048xbf16>) -> tensor<1x512x8192xbf16>
        %1307 = "ttir.multiply"(%1305, %1306) : (tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>) -> tensor<1x512x8192xbf16>
        %1308 = "ttir.linear"(%1307, %arg142) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16>, tensor<2048x8192xbf16>) -> tensor<1x512x2048xbf16>
        %1309 = "ttir.add"(%1293, %1308) : (tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1310 = "ttir.typecast"(%1309) <{conservative_folding = false}> : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
        %1311 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1312 = "ttir.pow"(%1310, %1311) : (tensor<1x512x2048xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
        %1313 = "ttir.mean"(%1312) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32>) -> tensor<1x512x1xf32>
        %1314 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
        %1315 = "ttir.add"(%1313, %1314) : (tensor<1x512x1xf32>, tensor<1xf32>) -> tensor<1x512x1xf32>
        %1316 = "ttir.rsqrt"(%1315) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1317 = "ttir.multiply"(%1310, %1316) : (tensor<1x512x2048xf32>, tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
        %1318 = "ttir.typecast"(%1317) <{conservative_folding = false}> : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
        %1319 = "ttir.multiply"(%arg145, %1318) : (tensor<2048xbf16>, tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
        %1320 = "ttir.linear"(%1319, %arg0) <{transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16>, tensor<128256x2048xbf16>) -> tensor<1x512x128256xbf16>
        %1321 = "ttir.slice_static"(%1320) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 511 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x128256xbf16>) -> tensor<1x511x128256xbf16>
        %1322 = "ttir.softmax"(%1321) <{dimension = 2 : si32, numericStable = true}> : (tensor<1x511x128256xbf16>) -> tensor<1x511x128256xbf16>
        %1323 = "ttir.multiply"(%1322, %arg149) : (tensor<1x511x128256xbf16>, tensor<1x511x128256xbf16>) -> tensor<1x511x128256xbf16>
        %1324 = "ttir.sum"(%1323) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x511x128256xbf16>) -> tensor<1x511x1xbf16>
        %1325 = "ttir.typecast"(%1324) <{conservative_folding = false}> : (tensor<1x511x1xbf16>) -> tensor<1x511x1xf32>
        %1326 = "ttir.clamp_scalar"(%1325) <{max = 0x7F800000 : f32, min = 9.99999996E-13 : f32}> : (tensor<1x511x1xf32>) -> tensor<1x511x1xf32>
        %1327 = "ttir.log"(%1326) : (tensor<1x511x1xf32>) -> tensor<1x511x1xf32>
        %1328 = "ttir.typecast"(%1327) <{conservative_folding = false}> : (tensor<1x511x1xf32>) -> tensor<1x511x1xbf16>
        %1329 = "ttir.typecast"(%1328) <{conservative_folding = false}> : (tensor<1x511x1xbf16>) -> tensor<1x511x1xf32>
        %1330 = "ttir.multiply"(%1329, %arg150) : (tensor<1x511x1xf32>, tensor<1x511x1xf32>) -> tensor<1x511x1xf32>
        %1331 = "ttir.sum"(%1330) <{dim_arg = [0 : i32, 1 : i32, 2 : i32], keep_dim = true}> : (tensor<1x511x1xf32>) -> tensor<1x1x1xf32>
        return %1331, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg147, %arg149, %arg150, %62, %68, %68, %70, %71, %81, %82, %108, %110, %113, %117, %117, %121, %124, %130, %130, %132, %133, %134, %135, %136, %137, %140, %146, %146, %148, %149, %159, %160, %186, %188, %191, %195, %195, %199, %202, %208, %208, %210, %211, %212, %213, %214, %215, %218, %224, %224, %226, %227, %237, %238, %264, %266, %269, %273, %273, %277, %280, %286, %286, %288, %289, %290, %291, %292, %293, %296, %302, %302, %304, %305, %315, %316, %342, %344, %347, %351, %351, %355, %358, %364, %364, %366, %367, %368, %369, %370, %371, %374, %380, %380, %382, %383, %393, %394, %420, %422, %425, %429, %429, %433, %436, %442, %442, %444, %445, %446, %447, %448, %449, %452, %458, %458, %460, %461, %471, %472, %498, %500, %503, %507, %507, %511, %514, %520, %520, %522, %523, %524, %525, %526, %527, %530, %536, %536, %538, %539, %549, %550, %576, %578, %581, %585, %585, %589, %592, %598, %598, %600, %601, %602, %603, %604, %605, %608, %614, %614, %616, %617, %627, %628, %654, %656, %659, %663, %663, %667, %670, %676, %676, %678, %679, %680, %681, %682, %683, %686, %692, %692, %694, %695, %705, %706, %732, %734, %737, %741, %741, %745, %748, %754, %754, %756, %757, %758, %759, %760, %761, %764, %770, %770, %772, %773, %783, %784, %810, %812, %815, %819, %819, %823, %826, %832, %832, %834, %835, %836, %837, %838, %839, %842, %848, %848, %850, %851, %861, %862, %888, %890, %893, %897, %897, %901, %904, %910, %910, %912, %913, %914, %915, %916, %917, %920, %926, %926, %928, %929, %939, %940, %966, %968, %971, %975, %975, %979, %982, %988, %988, %990, %991, %992, %993, %994, %995, %998, %1004, %1004, %1006, %1007, %1017, %1018, %1044, %1046, %1049, %1053, %1053, %1057, %1060, %1066, %1066, %1068, %1069, %1070, %1071, %1072, %1073, %1076, %1082, %1082, %1084, %1085, %1095, %1096, %1122, %1124, %1127, %1131, %1131, %1135, %1138, %1144, %1144, %1146, %1147, %1148, %1149, %1150, %1151, %1154, %1160, %1160, %1162, %1163, %1173, %1174, %1200, %1202, %1205, %1209, %1209, %1213, %1216, %1222, %1222, %1224, %1225, %1226, %1227, %1228, %1229, %1232, %1238, %1238, %1240, %1241, %1251, %1252, %1278, %1280, %1283, %1287, %1287, %1291, %1294, %1300, %1300, %1302, %1303, %1304, %1305, %1306, %1307, %1310, %1316, %1316, %1318, %1319, %1322, %1325, %1326 : tensor<1x1x1xf32>, tensor<128256x2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048x2048xbf16>, tensor<512x2048xbf16>, tensor<512x2048xbf16>, tensor<2048x2048xbf16>, tensor<8192x2048xbf16>, tensor<8192x2048xbf16>, tensor<2048x8192xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<2048xbf16>, tensor<1x512xi64>, tensor<1x511x128256xbf16>, tensor<1x511x1xf32>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x1x512x64xbf16>, tensor<1x1x512x64xbf16>, tensor<1x32x512x64xf32>, tensor<1x32x512x64xf32>, tensor<1x32x64x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x8192xbf16>, tensor<1x512x2048xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x2048xbf16>, tensor<1x512x2048xbf16>, tensor<1x511x128256xbf16>, tensor<1x511x1xf32>, tensor<1x511x1xf32>
      }
    }
  }
}
