// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,8 optimization-level=2" %s -o %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Single layer Llama 3.1 8B decode test extracted from full model
// Layer 0 with 8-chip tensor parallelism

#loc = loc(unknown)
ttcore.device_module {
  builtin.module @LlamaSingleLayerDecode  {
    // Single layer function with only layer 0 weights
    func.func @main(
      // Position index
      %arg0: tensor<1xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<1xi64>>, ttir.name = "position_ids"} loc("p0"),
      // Rotary embedding inverse frequencies
      %arg1: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<64xbf16>>, ttir.name = "rotary_emb_inv_freq"} loc("p1"),
      // Token indices
      %arg2: tensor<32x1xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<32x1xi64>>, ttir.name = "input_ids"} loc("p2"),
      // Embedding weights
      %arg3: tensor<128256x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<128256x4096xbf16>>, ttir.name = "embed_tokens_weight"} loc("p3"),
      // Layer 0 attention weights (pre-sharded for 8 chips with GQA)
      %arg4: tensor<1024x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<128x4096xbf16>>, ttir.name = "layer0_k_proj_weight"} loc("p4"),
      %arg5: tensor<1024x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<128x4096xbf16>>, ttir.name = "layer0_v_proj_weight"} loc("p5"),
      %arg6: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<512x4096xbf16>>, ttir.name = "layer0_q_proj_weight"} loc("p6"),
      %arg7: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<4096x512xbf16>>, ttir.name = "layer0_o_proj_weight"} loc("p7"),
      // Layer 0 KV cache inputs
      %arg8: tensor<32x8x128x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<32x1x128x128xbf16>>, ttir.name = "layer0_k_cache"} loc("p8"),
      %arg9: tensor<32x8x128x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<32x1x128x128xbf16>>, ttir.name = "layer0_v_cache"} loc("p9"),
      // Layer 0 norm weights
      %arg10: tensor<4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<4096xbf16>>, ttir.name = "layer0_input_layernorm_weight"} loc("p10"),
      %arg11: tensor<4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<4096xbf16>>, ttir.name = "layer0_post_attn_layernorm_weight"} loc("p11"),
      // Layer 0 FFN weights (sharded for 8 chips: 14336/8 = 1792)
      %arg12: tensor<14336x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<1792x4096xbf16>>, ttir.name = "layer0_gate_proj_weight"} loc("p12"),
      %arg13: tensor<14336x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<1792x4096xbf16>>, ttir.name = "layer0_up_proj_weight"} loc("p13"),
      %arg14: tensor<4096x14336xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<4096x1792xbf16>>, ttir.name = "layer0_down_proj_weight"} loc("p14")
    ) -> (tensor<32x1x4096xbf16>, tensor<32x8x128x128xbf16>, tensor<32x8x128x128xbf16>) {

      // === Mesh shard inputs for 8-chip tensor parallelism ===
      %pos = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1xi64>) -> tensor<1xi64> loc(#loc)
      %inv_freq = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<64xbf16>) -> tensor<64xbf16> loc(#loc)
      %input_ids = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x1xi64>) -> tensor<32x1xi64> loc(#loc)
      %embed_weight = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128256x4096xbf16>) -> tensor<128256x4096xbf16> loc(#loc)

      // Attention weights sharded across head dimension (8 KV heads -> 1 per chip)
      %k_proj = "ttir.mesh_shard"(%arg4) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1024x4096xbf16>) -> tensor<128x4096xbf16> loc(#loc)
      %v_proj = "ttir.mesh_shard"(%arg5) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1024x4096xbf16>) -> tensor<128x4096xbf16> loc(#loc)
      %q_proj = "ttir.mesh_shard"(%arg6) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096x4096xbf16>) -> tensor<512x4096xbf16> loc(#loc)
      %o_proj = "ttir.mesh_shard"(%arg7) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 8>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096x4096xbf16>) -> tensor<4096x512xbf16> loc(#loc)

      // KV cache sharded across head dimension
      %k_cache = "ttir.mesh_shard"(%arg8) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 8, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x8x128x128xbf16>) -> tensor<32x1x128x128xbf16> loc(#loc)
      %v_cache = "ttir.mesh_shard"(%arg9) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 8, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x8x128x128xbf16>) -> tensor<32x1x128x128xbf16> loc(#loc)

      // Norm weights (replicated)
      %input_ln_w = "ttir.mesh_shard"(%arg10) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096xbf16>) -> tensor<4096xbf16> loc(#loc)
      %post_attn_ln_w = "ttir.mesh_shard"(%arg11) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096xbf16>) -> tensor<4096xbf16> loc(#loc)

      // FFN weights sharded (14336/8 = 1792 per chip)
      %gate_proj = "ttir.mesh_shard"(%arg12) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<14336x4096xbf16>) -> tensor<1792x4096xbf16> loc(#loc)
      %up_proj = "ttir.mesh_shard"(%arg13) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<14336x4096xbf16>) -> tensor<1792x4096xbf16> loc(#loc)
      %down_proj = "ttir.mesh_shard"(%arg14) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 8>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096x14336xbf16>) -> tensor<4096x1792xbf16> loc(#loc)

      // === Constants ===
      %c0_i64 = "ttir.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64> loc(#loc)
      %c128_i64 = "ttir.constant"() <{value = dense<128> : tensor<1xi64>}> : () -> tensor<1xi64> loc(#loc)
      %c2_f32 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32> loc(#loc)
      %c_inv_hidden = "ttir.constant"() <{value = dense<2.44140625E-4> : tensor<f32>}> : () -> tensor<f32> loc(#loc)  // 1/4096
      %c_eps = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32> loc(#loc)  // RMSNorm epsilon
      %c_scale = "ttir.constant"() <{value = dense<0.297301769> : tensor<f32>}> : () -> tensor<f32> loc(#loc)  // 1/sqrt(head_dim) = 1/sqrt(128) ≈ 0.0884, but scaled for GQA
      %c0_bf16 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16> loc(#loc)
      %c_neg_inf_bf16 = "ttir.constant"() <{value = dense<0xFF80> : tensor<bf16>}> : () -> tensor<bf16> loc(#loc)
      %c_neg_inf_f64 = "ttir.constant"() <{value = dense<0xFFF0000000000000> : tensor<f64>}> : () -> tensor<f64> loc(#loc)

      // Position indices for causal mask
      %pos_indices = "ttir.constant"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F00000000000000"> : tensor<1x128xi64>}> : () -> tensor<1x128xi64> loc(#loc)
      %c0_f32 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32> loc(#loc)

      // === Broadcast constants to needed shapes ===
      %zeros_attn_4d = "ttir.reshape"(%c0_f32) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32> loc(#loc)
      %zeros_attn = "ttir.broadcast"(%zeros_attn_4d) <{broadcast_dimensions = array<i64: 32, 4, 1, 128>}> : (tensor<1x1x1x1xf32>) -> tensor<32x4x1x128xf32> loc(#loc)

      %neg_inf_f64_4d = "ttir.reshape"(%c_neg_inf_f64) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f64>) -> tensor<1x1x1x1xf64> loc(#loc)
      %neg_inf_attn = "ttir.broadcast"(%neg_inf_f64_4d) <{broadcast_dimensions = array<i64: 32, 4, 1, 128>}> : (tensor<1x1x1x1xf64>) -> tensor<32x4x1x128xf64> loc(#loc)

      %neg_inf_bf16_4d = "ttir.reshape"(%c_neg_inf_bf16) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16> loc(#loc)
      %neg_inf_mask = "ttir.broadcast"(%neg_inf_bf16_4d) <{broadcast_dimensions = array<i64: 32, 1, 1, 128>}> : (tensor<1x1x1x1xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)

      %zeros_bf16_4d = "ttir.reshape"(%c0_bf16) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16> loc(#loc)
      %zeros_mask = "ttir.broadcast"(%zeros_bf16_4d) <{broadcast_dimensions = array<i64: 32, 1, 1, 128>}> : (tensor<1x1x1x1xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)

      %scale_4d = "ttir.reshape"(%c_scale) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32> loc(#loc)
      %scale_kv = "ttir.broadcast"(%scale_4d) <{broadcast_dimensions = array<i64: 32, 4, 128, 128>}> : (tensor<1x1x1x1xf32>) -> tensor<32x4x128x128xf32> loc(#loc)
      %scale_q = "ttir.broadcast"(%scale_4d) <{broadcast_dimensions = array<i64: 32, 4, 1, 128>}> : (tensor<1x1x1x1xf32>) -> tensor<32x4x1x128xf32> loc(#loc)

      %eps_3d = "ttir.reshape"(%c_eps) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32> loc(#loc)
      %eps = "ttir.broadcast"(%eps_3d) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x1xf32>) -> tensor<32x1x1xf32> loc(#loc)

      %inv_hidden_2d = "ttir.reshape"(%c_inv_hidden) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1xf32> loc(#loc)
      %inv_hidden = "ttir.broadcast"(%inv_hidden_2d) <{broadcast_dimensions = array<i64: 32, 1>}> : (tensor<1x1xf32>) -> tensor<32x1xf32> loc(#loc)

      %pow_exp_3d = "ttir.reshape"(%c2_f32) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32> loc(#loc)
      %pow_exp = "ttir.broadcast"(%pow_exp_3d) <{broadcast_dimensions = array<i64: 32, 1, 4096>}> : (tensor<1x1x1xf32>) -> tensor<32x1x4096xf32> loc(#loc)

      // === Prepare position for causal mask ===
      %pos_reshaped = "ttir.reshape"(%pos) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi64>) -> tensor<1x1x1xi64> loc(#loc)
      %pos_flat = "ttir.reshape"(%pos_reshaped) <{shape = [1 : i32]}> : (tensor<1x1x1xi64>) -> tensor<1xi64> loc(#loc)
      %pos_lt_zero = "ttir.lt"(%pos_flat, %c0_i64) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1> loc(#loc)
      %pos_plus_max = "ttir.add"(%pos_flat, %c128_i64) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64> loc(#loc)
      %pos_normalized = "ttir.where"(%pos_lt_zero, %pos_plus_max, %pos_flat) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64> loc(#loc)

      // === Embedding lookup ===
      %input_ln_w_3d = "ttir.reshape"(%input_ln_w) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16> loc(#loc)
      %input_ln_w_bc = "ttir.broadcast"(%input_ln_w_3d) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x4096xbf16>) -> tensor<32x1x4096xbf16> loc(#loc)

      %embed_2d = "ttir.reshape"(%embed_weight) <{shape = [128256 : i32, 4096 : i32]}> : (tensor<128256x4096xbf16>) -> tensor<128256x4096xbf16> loc(#loc)
      %indices = "ttir.reshape"(%input_ids) <{shape = [32 : i32]}> : (tensor<32x1xi64>) -> tensor<32xi64> loc(#loc)
      %indices_u32 = "ttir.typecast"(%indices) <{conservative_folding = false}> : (tensor<32xi64>) -> tensor<32xui32> loc(#loc)

      %embedded = "ttir.gather"(%embed_2d, %indices_u32) <{
        collapsed_slice_dims = array<i64: 0>,
        index_vector_dim = 1 : si64,
        indices_are_sorted = false,
        offset_dims = array<i64: 1>,
        operand_batching_dims = array<i64>,
        slice_sizes = array<i64: 1, 4096>,
        start_index_map = array<i64: 0>,
        start_indices_batching_dims = array<i64>
      }> : (tensor<128256x4096xbf16>, tensor<32xui32>) -> tensor<32x4096xbf16> loc(#loc)

      %hidden_3d = "ttir.reshape"(%embedded) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<32x4096xbf16>) -> tensor<32x1x4096xbf16> loc(#loc)

      // === RMSNorm before attention ===
      %hidden_f32 = "ttir.typecast"(%hidden_3d) <{conservative_folding = false}> : (tensor<32x1x4096xbf16>) -> tensor<32x1x4096xf32> loc(#loc)
      %hidden_sq = "ttir.pow"(%hidden_f32, %pow_exp) : (tensor<32x1x4096xf32>, tensor<32x1x4096xf32>) -> tensor<32x1x4096xf32> loc(#loc)
      %variance = "ttir.sum"(%hidden_sq) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x4096xf32>) -> tensor<32x1xf32> loc(#loc)
      %mean_sq = "ttir.multiply"(%variance, %inv_hidden) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32> loc(#loc)
      %mean_sq_3d = "ttir.reshape"(%mean_sq) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32> loc(#loc)
      %mean_sq_eps = "ttir.add"(%mean_sq_3d, %eps) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32> loc(#loc)
      %rms_inv = "ttir.rsqrt"(%mean_sq_eps) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32> loc(#loc)
      %rms_inv_bc = "ttir.broadcast"(%rms_inv) <{broadcast_dimensions = array<i64: 1, 1, 4096>}> : (tensor<32x1x1xf32>) -> tensor<32x1x4096xf32> loc(#loc)
      %hidden_normed_f32 = "ttir.multiply"(%hidden_f32, %rms_inv_bc) : (tensor<32x1x4096xf32>, tensor<32x1x4096xf32>) -> tensor<32x1x4096xf32> loc(#loc)
      %hidden_normed = "ttir.typecast"(%hidden_normed_f32) <{conservative_folding = false}> : (tensor<32x1x4096xf32>) -> tensor<32x1x4096xbf16> loc(#loc)
      %attn_input = "ttir.multiply"(%input_ln_w_bc, %hidden_normed) : (tensor<32x1x4096xbf16>, tensor<32x1x4096xbf16>) -> tensor<32x1x4096xbf16> loc(#loc)
      %attn_input_2d = "ttir.reshape"(%attn_input) <{shape = [32 : i32, 4096 : i32]}> : (tensor<32x1x4096xbf16>) -> tensor<32x4096xbf16> loc(#loc)

      // === Q/K/V Projections ===
      // K projection (128 heads total, sharded -> 1 head per chip = 128 dim)
      %k_proj_t = "ttir.permute"(%k_proj) <{permutation = array<i64: 1, 0>}> : (tensor<128x4096xbf16>) -> tensor<4096x128xbf16> loc(#loc)
      %k_out = "ttir.dot_general"(%attn_input_2d, %k_proj_t) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x128xbf16>) -> tensor<32x128xbf16> loc(#loc)
      %k_4d = "ttir.reshape"(%k_out) <{shape = [32 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<32x128xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)

      // V projection
      %v_proj_t = "ttir.permute"(%v_proj) <{permutation = array<i64: 1, 0>}> : (tensor<128x4096xbf16>) -> tensor<4096x128xbf16> loc(#loc)
      %v_out = "ttir.dot_general"(%attn_input_2d, %v_proj_t) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x128xbf16>) -> tensor<32x128xbf16> loc(#loc)
      %v_4d = "ttir.reshape"(%v_out) <{shape = [32 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<32x128xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)

      // Q projection (4 Q heads per 1 KV head = GQA with 4x ratio)
      %q_proj_t = "ttir.permute"(%q_proj) <{permutation = array<i64: 1, 0>}> : (tensor<512x4096xbf16>) -> tensor<4096x512xbf16> loc(#loc)
      %q_out = "ttir.dot_general"(%attn_input_2d, %q_proj_t) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x512xbf16>) -> tensor<32x512xbf16> loc(#loc)
      %q_4d = "ttir.reshape"(%q_out) <{shape = [32 : i32, 4 : i32, 1 : i32, 128 : i32]}> : (tensor<32x512xbf16>) -> tensor<32x4x1x128xbf16> loc(#loc)

      // === RoPE (Rotary Position Embeddings) ===
      %inv_freq_3d = "ttir.reshape"(%inv_freq) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<64xbf16>) -> tensor<1x64x1xbf16> loc(#loc)
      %inv_freq_f32 = "ttir.typecast"(%inv_freq_3d) <{conservative_folding = false}> : (tensor<1x64x1xbf16>) -> tensor<1x64x1xf32> loc(#loc)
      %pos_f32 = "ttir.typecast"(%pos_reshaped) <{conservative_folding = false}> : (tensor<1x1x1xi64>) -> tensor<1x1x1xf32> loc(#loc)
      %freqs = "ttir.dot_general"(%inv_freq_f32, %pos_f32) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<1x64x1xf32>, tensor<1x1x1xf32>) -> tensor<1x64x1xf32> loc(#loc)
      %freqs_reshaped = "ttir.reshape"(%freqs) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64x1xf32>) -> tensor<1x1x64xf32> loc(#loc)
      %freqs_doubled = "ttir.concat"(%freqs_reshaped, %freqs_reshaped) <{dim = 2 : si32}> : (tensor<1x1x64xf32>, tensor<1x1x64xf32>) -> tensor<1x1x128xf32> loc(#loc)

      %cos_freqs = "ttir.cos"(%freqs_doubled) : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32> loc(#loc)
      %cos_bf16 = "ttir.typecast"(%cos_freqs) <{conservative_folding = false}> : (tensor<1x1x128xf32>) -> tensor<1x1x128xbf16> loc(#loc)
      %cos_4d = "ttir.reshape"(%cos_bf16) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<1x1x1x128xbf16> loc(#loc)
      %cos_k_bc = "ttir.broadcast"(%cos_4d) <{broadcast_dimensions = array<i64: 32, 1, 1, 1>}> : (tensor<1x1x1x128xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)
      %cos_q_bc = "ttir.broadcast"(%cos_4d) <{broadcast_dimensions = array<i64: 32, 4, 1, 1>}> : (tensor<1x1x1x128xbf16>) -> tensor<32x4x1x128xbf16> loc(#loc)

      %sin_freqs = "ttir.sin"(%freqs_doubled) : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32> loc(#loc)
      %sin_bf16 = "ttir.typecast"(%sin_freqs) <{conservative_folding = false}> : (tensor<1x1x128xf32>) -> tensor<1x1x128xbf16> loc(#loc)
      %sin_4d = "ttir.reshape"(%sin_bf16) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<1x1x1x128xbf16> loc(#loc)
      %sin_k_bc = "ttir.broadcast"(%sin_4d) <{broadcast_dimensions = array<i64: 32, 1, 1, 1>}> : (tensor<1x1x1x128xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)
      %sin_q_bc = "ttir.broadcast"(%sin_4d) <{broadcast_dimensions = array<i64: 32, 4, 1, 1>}> : (tensor<1x1x1x128xbf16>) -> tensor<32x4x1x128xbf16> loc(#loc)

      // Apply RoPE to K
      %k_cos = "ttir.multiply"(%k_4d, %cos_k_bc) : (tensor<32x1x1x128xbf16>, tensor<32x1x1x128xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)
      %k_hi = "ttir.slice_static"(%k_4d) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [32 : i32, 1 : i32, 1 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x1x128xbf16>) -> tensor<32x1x1x64xbf16> loc(#loc)
      %k_hi_neg = "ttir.neg"(%k_hi) : (tensor<32x1x1x64xbf16>) -> tensor<32x1x1x64xbf16> loc(#loc)
      %k_lo = "ttir.slice_static"(%k_4d) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 1 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1x1x128xbf16>) -> tensor<32x1x1x64xbf16> loc(#loc)
      %k_rotated = "ttir.concat"(%k_hi_neg, %k_lo) <{dim = 3 : si32}> : (tensor<32x1x1x64xbf16>, tensor<32x1x1x64xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)
      %k_sin = "ttir.multiply"(%k_rotated, %sin_k_bc) : (tensor<32x1x1x128xbf16>, tensor<32x1x1x128xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)
      %k_rope = "ttir.add"(%k_cos, %k_sin) : (tensor<32x1x1x128xbf16>, tensor<32x1x1x128xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)

      // Apply RoPE to Q
      %q_cos = "ttir.multiply"(%q_4d, %cos_q_bc) : (tensor<32x4x1x128xbf16>, tensor<32x4x1x128xbf16>) -> tensor<32x4x1x128xbf16> loc(#loc)
      %q_hi = "ttir.slice_static"(%q_4d) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [32 : i32, 4 : i32, 1 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x4x1x128xbf16>) -> tensor<32x4x1x64xbf16> loc(#loc)
      %q_hi_neg = "ttir.neg"(%q_hi) : (tensor<32x4x1x64xbf16>) -> tensor<32x4x1x64xbf16> loc(#loc)
      %q_lo = "ttir.slice_static"(%q_4d) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 4 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x4x1x128xbf16>) -> tensor<32x4x1x64xbf16> loc(#loc)
      %q_rotated = "ttir.concat"(%q_hi_neg, %q_lo) <{dim = 3 : si32}> : (tensor<32x4x1x64xbf16>, tensor<32x4x1x64xbf16>) -> tensor<32x4x1x128xbf16> loc(#loc)
      %q_sin = "ttir.multiply"(%q_rotated, %sin_q_bc) : (tensor<32x4x1x128xbf16>, tensor<32x4x1x128xbf16>) -> tensor<32x4x1x128xbf16> loc(#loc)
      %q_rope = "ttir.add"(%q_cos, %q_sin) : (tensor<32x4x1x128xbf16>, tensor<32x4x1x128xbf16>) -> tensor<32x4x1x128xbf16> loc(#loc)

      // === Update KV Cache ===
      %k_for_cache = "ttir.permute"(%k_rope) <{permutation = array<i64: 2, 1, 0, 3>}> : (tensor<32x1x1x128xbf16>) -> tensor<1x1x32x128xbf16> loc(#loc)
      %k_cache_updated = "ttir.update_cache"(%k_cache, %k_for_cache, %pos) <{batch_offset = 0 : i32}> : (tensor<32x1x128x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1xi64>) -> tensor<32x1x128x128xbf16> loc(#loc)

      %v_for_cache = "ttir.permute"(%v_4d) <{permutation = array<i64: 2, 1, 0, 3>}> : (tensor<32x1x1x128xbf16>) -> tensor<1x1x32x128xbf16> loc(#loc)
      %v_cache_updated = "ttir.update_cache"(%v_cache, %v_for_cache, %pos) <{batch_offset = 0 : i32}> : (tensor<32x1x128x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1xi64>) -> tensor<32x1x128x128xbf16> loc(#loc)

      // === Attention computation ===
      // Scale Q
      %q_f32 = "ttir.typecast"(%q_rope) <{conservative_folding = false}> : (tensor<32x4x1x128xbf16>) -> tensor<32x4x1x128xf32> loc(#loc)
      %q_scaled = "ttir.multiply"(%q_f32, %scale_q) : (tensor<32x4x1x128xf32>, tensor<32x4x1x128xf32>) -> tensor<32x4x1x128xf32> loc(#loc)

      // Expand K cache for GQA (1 KV head -> 4 Q heads)
      %k_cache_5d = "ttir.reshape"(%k_cache_updated) <{shape = [32 : i32, 1 : i32, 1 : i32, 128 : i32, 128 : i32]}> : (tensor<32x1x128x128xbf16>) -> tensor<32x1x1x128x128xbf16> loc(#loc)
      %k_cache_expanded = "ttir.broadcast"(%k_cache_5d) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x1x1x128x128xbf16>) -> tensor<32x1x4x128x128xbf16> loc(#loc)
      %k_cache_4d = "ttir.reshape"(%k_cache_expanded) <{shape = [32 : i32, 4 : i32, 128 : i32, 128 : i32]}> : (tensor<32x1x4x128x128xbf16>) -> tensor<32x4x128x128xbf16> loc(#loc)
      %k_cache_f32 = "ttir.typecast"(%k_cache_4d) <{conservative_folding = false}> : (tensor<32x4x128x128xbf16>) -> tensor<32x4x128x128xf32> loc(#loc)
      %k_cache_t = "ttir.permute"(%k_cache_f32) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<32x4x128x128xf32>) -> tensor<32x4x128x128xf32> loc(#loc)
      %k_scaled = "ttir.multiply"(%k_cache_t, %scale_kv) : (tensor<32x4x128x128xf32>, tensor<32x4x128x128xf32>) -> tensor<32x4x128x128xf32> loc(#loc)

      // QK^T
      %attn_weights = "ttir.dot_general"(%q_scaled, %k_scaled) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<32x4x1x128xf32>, tensor<32x4x128x128xf32>) -> tensor<32x4x1x128xf32> loc(#loc)

      // === Causal mask ===
      %pos_bc = "ttir.reshape"(%pos_normalized) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi64>) -> tensor<1x1xi64> loc(#loc)
      %pos_bc_128 = "ttir.broadcast"(%pos_bc) <{broadcast_dimensions = array<i64: 1, 128>}> : (tensor<1x1xi64>) -> tensor<1x128xi64> loc(#loc)
      %causal_mask_1d = "ttir.le"(%pos_indices, %pos_bc_128) : (tensor<1x128xi64>, tensor<1x128xi64>) -> tensor<1x128xi1> loc(#loc)
      %causal_mask_3d = "ttir.reshape"(%causal_mask_1d) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xi1>) -> tensor<1x1x128xi1> loc(#loc)
      %causal_mask_4d = "ttir.reshape"(%causal_mask_3d) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x1x128xi1>) -> tensor<1x1x1x128xi1> loc(#loc)
      %causal_mask = "ttir.broadcast"(%causal_mask_4d) <{broadcast_dimensions = array<i64: 32, 1, 1, 1>}> : (tensor<1x1x1x128xi1>) -> tensor<32x1x1x128xi1> loc(#loc)

      // Apply mask: where(mask, 0, -inf)
      %mask_bias = "ttir.where"(%causal_mask, %zeros_mask, %neg_inf_mask) : (tensor<32x1x1x128xi1>, tensor<32x1x1x128xbf16>, tensor<32x1x1x128xbf16>) -> tensor<32x1x1x128xbf16> loc(#loc)
      %mask_bias_f32 = "ttir.typecast"(%mask_bias) <{conservative_folding = false}> : (tensor<32x1x1x128xbf16>) -> tensor<32x1x1x128xf32> loc(#loc)
      %mask_bias_4h = "ttir.broadcast"(%mask_bias_f32) <{broadcast_dimensions = array<i64: 1, 4, 1, 1>}> : (tensor<32x1x1x128xf32>) -> tensor<32x4x1x128xf32> loc(#loc)

      %attn_masked = "ttir.add"(%attn_weights, %mask_bias_4h) : (tensor<32x4x1x128xf32>, tensor<32x4x1x128xf32>) -> tensor<32x4x1x128xf32> loc(#loc)

      // === Softmax ===
      // Check for all-masked rows
      %attn_f64 = "ttir.typecast"(%attn_masked) <{conservative_folding = false}> : (tensor<32x4x1x128xf32>) -> tensor<32x4x1x128xf64> loc(#loc)
      %is_neg_inf = "ttir.eq"(%attn_f64, %neg_inf_attn) : (tensor<32x4x1x128xf64>, tensor<32x4x1x128xf64>) -> tensor<32x4x1x128xi1> loc(#loc)
      %is_not_neg_inf = "ttir.logical_not"(%is_neg_inf) : (tensor<32x4x1x128xi1>) -> tensor<32x4x1x128xi1> loc(#loc)
      %any_valid = "ttir.reduce_or"(%is_not_neg_inf) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x4x1x128xi1>) -> tensor<32x4x1xi1> loc(#loc)
      %all_masked = "ttir.logical_not"(%any_valid) : (tensor<32x4x1xi1>) -> tensor<32x4x1xi1> loc(#loc)
      %all_masked_4d = "ttir.reshape"(%all_masked) <{shape = [32 : i32, 4 : i32, 1 : i32, 1 : i32]}> : (tensor<32x4x1xi1>) -> tensor<32x4x1x1xi1> loc(#loc)
      %all_masked_bc = "ttir.broadcast"(%all_masked_4d) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x4x1x1xi1>) -> tensor<32x4x1x128xi1> loc(#loc)

      // Stable softmax: exp(x - max) / sum(exp(x - max))
      %attn_max = "ttir.max"(%attn_masked) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x4x1x128xf32>) -> tensor<32x4x1xf32> loc(#loc)
      %attn_max_4d = "ttir.reshape"(%attn_max) <{shape = [32 : i32, 4 : i32, 1 : i32, 1 : i32]}> : (tensor<32x4x1xf32>) -> tensor<32x4x1x1xf32> loc(#loc)
      %attn_max_bc = "ttir.broadcast"(%attn_max_4d) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x4x1x1xf32>) -> tensor<32x4x1x128xf32> loc(#loc)
      %attn_shifted = "ttir.subtract"(%attn_masked, %attn_max_bc) : (tensor<32x4x1x128xf32>, tensor<32x4x1x128xf32>) -> tensor<32x4x1x128xf32> loc(#loc)
      %attn_exp = "ttir.exp"(%attn_shifted) : (tensor<32x4x1x128xf32>) -> tensor<32x4x1x128xf32> loc(#loc)
      %attn_sum = "ttir.sum"(%attn_exp) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x4x1x128xf32>) -> tensor<32x4x1xf32> loc(#loc)
      %attn_sum_4d = "ttir.reshape"(%attn_sum) <{shape = [32 : i32, 4 : i32, 1 : i32, 1 : i32]}> : (tensor<32x4x1xf32>) -> tensor<32x4x1x1xf32> loc(#loc)
      %attn_sum_bc = "ttir.broadcast"(%attn_sum_4d) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x4x1x1xf32>) -> tensor<32x4x1x128xf32> loc(#loc)
      %attn_softmax = "ttir.div"(%attn_exp, %attn_sum_bc) : (tensor<32x4x1x128xf32>, tensor<32x4x1x128xf32>) -> tensor<32x4x1x128xf32> loc(#loc)

      // Zero out fully masked rows
      %attn_probs = "ttir.where"(%all_masked_bc, %zeros_attn, %attn_softmax) : (tensor<32x4x1x128xi1>, tensor<32x4x1x128xf32>, tensor<32x4x1x128xf32>) -> tensor<32x4x1x128xf32> loc(#loc)

      // === Attention @ V ===
      %v_cache_5d = "ttir.reshape"(%v_cache_updated) <{shape = [32 : i32, 1 : i32, 1 : i32, 128 : i32, 128 : i32]}> : (tensor<32x1x128x128xbf16>) -> tensor<32x1x1x128x128xbf16> loc(#loc)
      %v_cache_expanded = "ttir.broadcast"(%v_cache_5d) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x1x1x128x128xbf16>) -> tensor<32x1x4x128x128xbf16> loc(#loc)
      %v_cache_4d = "ttir.reshape"(%v_cache_expanded) <{shape = [32 : i32, 4 : i32, 128 : i32, 128 : i32]}> : (tensor<32x1x4x128x128xbf16>) -> tensor<32x4x128x128xbf16> loc(#loc)
      %v_cache_f32 = "ttir.typecast"(%v_cache_4d) <{conservative_folding = false}> : (tensor<32x4x128x128xbf16>) -> tensor<32x4x128x128xf32> loc(#loc)

      %attn_out = "ttir.dot_general"(%attn_probs, %v_cache_f32) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<32x4x1x128xf32>, tensor<32x4x128x128xf32>) -> tensor<32x4x1x128xf32> loc(#loc)
      %attn_out_bf16 = "ttir.typecast"(%attn_out) <{conservative_folding = false}> : (tensor<32x4x1x128xf32>) -> tensor<32x4x1x128xbf16> loc(#loc)
      %attn_out_2d = "ttir.reshape"(%attn_out_bf16) <{shape = [32 : i32, 512 : i32]}> : (tensor<32x4x1x128xbf16>) -> tensor<32x512xbf16> loc(#loc)

      // === Output projection ===
      %o_proj_t = "ttir.permute"(%o_proj) <{permutation = array<i64: 1, 0>}> : (tensor<4096x512xbf16>) -> tensor<512x4096xbf16> loc(#loc)
      %attn_proj = "ttir.dot_general"(%attn_out_2d, %o_proj_t) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x512xbf16>, tensor<512x4096xbf16>) -> tensor<32x4096xbf16> loc(#loc)
      %attn_proj_reduced = "ttir.all_reduce"(%attn_proj) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x4096xbf16>) -> tensor<32x4096xbf16> loc(#loc)
      %attn_proj_3d = "ttir.reshape"(%attn_proj_reduced) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<32x4096xbf16>) -> tensor<32x1x4096xbf16> loc(#loc)

      // === First residual add ===
      %hidden_post_attn = "ttir.add"(%hidden_3d, %attn_proj_3d) : (tensor<32x1x4096xbf16>, tensor<32x1x4096xbf16>) -> tensor<32x1x4096xbf16> loc(#loc)

      // === RMSNorm before FFN ===
      %post_attn_ln_w_3d = "ttir.reshape"(%post_attn_ln_w) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16> loc(#loc)
      %post_attn_ln_w_bc = "ttir.broadcast"(%post_attn_ln_w_3d) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x4096xbf16>) -> tensor<32x1x4096xbf16> loc(#loc)

      %hidden2_f32 = "ttir.typecast"(%hidden_post_attn) <{conservative_folding = false}> : (tensor<32x1x4096xbf16>) -> tensor<32x1x4096xf32> loc(#loc)
      %hidden2_sq = "ttir.pow"(%hidden2_f32, %pow_exp) : (tensor<32x1x4096xf32>, tensor<32x1x4096xf32>) -> tensor<32x1x4096xf32> loc(#loc)
      %variance2 = "ttir.sum"(%hidden2_sq) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x4096xf32>) -> tensor<32x1xf32> loc(#loc)
      %mean_sq2 = "ttir.multiply"(%variance2, %inv_hidden) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32> loc(#loc)
      %mean_sq2_3d = "ttir.reshape"(%mean_sq2) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32> loc(#loc)
      %mean_sq2_eps = "ttir.add"(%mean_sq2_3d, %eps) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32> loc(#loc)
      %rms_inv2 = "ttir.rsqrt"(%mean_sq2_eps) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32> loc(#loc)
      %rms_inv2_bc = "ttir.broadcast"(%rms_inv2) <{broadcast_dimensions = array<i64: 1, 1, 4096>}> : (tensor<32x1x1xf32>) -> tensor<32x1x4096xf32> loc(#loc)
      %hidden2_normed_f32 = "ttir.multiply"(%hidden2_f32, %rms_inv2_bc) : (tensor<32x1x4096xf32>, tensor<32x1x4096xf32>) -> tensor<32x1x4096xf32> loc(#loc)
      %hidden2_normed = "ttir.typecast"(%hidden2_normed_f32) <{conservative_folding = false}> : (tensor<32x1x4096xf32>) -> tensor<32x1x4096xbf16> loc(#loc)
      %ffn_input = "ttir.multiply"(%post_attn_ln_w_bc, %hidden2_normed) : (tensor<32x1x4096xbf16>, tensor<32x1x4096xbf16>) -> tensor<32x1x4096xbf16> loc(#loc)
      %ffn_input_2d = "ttir.reshape"(%ffn_input) <{shape = [32 : i32, 4096 : i32]}> : (tensor<32x1x4096xbf16>) -> tensor<32x4096xbf16> loc(#loc)

      // === FFN (SwiGLU) ===
      // Gate projection
      %gate_proj_t = "ttir.permute"(%gate_proj) <{permutation = array<i64: 1, 0>}> : (tensor<1792x4096xbf16>) -> tensor<4096x1792xbf16> loc(#loc)
      %gate_out = "ttir.dot_general"(%ffn_input_2d, %gate_proj_t) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x1792xbf16>) -> tensor<32x1792xbf16> loc(#loc)
      %gate_out_3d = "ttir.reshape"(%gate_out) <{shape = [32 : i32, 1 : i32, 1792 : i32]}> : (tensor<32x1792xbf16>) -> tensor<32x1x1792xbf16> loc(#loc)

      // SiLU activation
      %gate_f32 = "ttir.typecast"(%gate_out_3d) <{conservative_folding = false}> : (tensor<32x1x1792xbf16>) -> tensor<32x1x1792xf32> loc(#loc)
      %gate_sigmoid = "ttir.sigmoid"(%gate_f32) : (tensor<32x1x1792xf32>) -> tensor<32x1x1792xf32> loc(#loc)
      %gate_silu = "ttir.multiply"(%gate_f32, %gate_sigmoid) : (tensor<32x1x1792xf32>, tensor<32x1x1792xf32>) -> tensor<32x1x1792xf32> loc(#loc)
      %gate_silu_bf16 = "ttir.typecast"(%gate_silu) <{conservative_folding = false}> : (tensor<32x1x1792xf32>) -> tensor<32x1x1792xbf16> loc(#loc)

      // Up projection
      %up_proj_t = "ttir.permute"(%up_proj) <{permutation = array<i64: 1, 0>}> : (tensor<1792x4096xbf16>) -> tensor<4096x1792xbf16> loc(#loc)
      %up_out = "ttir.dot_general"(%ffn_input_2d, %up_proj_t) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x1792xbf16>) -> tensor<32x1792xbf16> loc(#loc)
      %up_out_3d = "ttir.reshape"(%up_out) <{shape = [32 : i32, 1 : i32, 1792 : i32]}> : (tensor<32x1792xbf16>) -> tensor<32x1x1792xbf16> loc(#loc)

      // Gate * Up
      %ffn_hidden = "ttir.multiply"(%gate_silu_bf16, %up_out_3d) : (tensor<32x1x1792xbf16>, tensor<32x1x1792xbf16>) -> tensor<32x1x1792xbf16> loc(#loc)
      %ffn_hidden_2d = "ttir.reshape"(%ffn_hidden) <{shape = [32 : i32, 1792 : i32]}> : (tensor<32x1x1792xbf16>) -> tensor<32x1792xbf16> loc(#loc)

      // Down projection
      %down_proj_t = "ttir.permute"(%down_proj) <{permutation = array<i64: 1, 0>}> : (tensor<4096x1792xbf16>) -> tensor<1792x4096xbf16> loc(#loc)
      %ffn_out = "ttir.dot_general"(%ffn_hidden_2d, %down_proj_t) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x1792xbf16>, tensor<1792x4096xbf16>) -> tensor<32x4096xbf16> loc(#loc)
      %ffn_out_reduced = "ttir.all_reduce"(%ffn_out) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x4096xbf16>) -> tensor<32x4096xbf16> loc(#loc)
      %ffn_out_3d = "ttir.reshape"(%ffn_out_reduced) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<32x4096xbf16>) -> tensor<32x1x4096xbf16> loc(#loc)

      // === Second residual add ===
      %output = "ttir.add"(%hidden_post_attn, %ffn_out_3d) : (tensor<32x1x4096xbf16>, tensor<32x1x4096xbf16>) -> tensor<32x1x4096xbf16> loc(#loc)

      // === Unshard KV caches for return ===
      %k_cache_out = "ttir.mesh_shard"(%k_cache_updated) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 8, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x1x128x128xbf16>) -> tensor<32x8x128x128xbf16> loc(#loc)
      %v_cache_out = "ttir.mesh_shard"(%v_cache_updated) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 8, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x1x128x128xbf16>) -> tensor<32x8x128x128xbf16> loc(#loc)

      return %output, %k_cache_out, %v_cache_out : tensor<32x1x4096xbf16>, tensor<32x8x128x128xbf16>, tensor<32x8x128x128xbf16> loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
