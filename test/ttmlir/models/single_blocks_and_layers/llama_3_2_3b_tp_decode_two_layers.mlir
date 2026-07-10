module @SyncTensorsGraph.6845 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.6845 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
      func.func @main(%arg0: tensor<1x1xi32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1xi32>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_3"}, %arg1: tensor<1xi32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1xi32>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_2"}, %arg2: tensor<64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<64xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_0_self_attn_rotary_emb_inv_freq"}, %arg3: tensor<1x1xi32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1xi32>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg4: tensor<512x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_0_self_attn_qkv_proj_v_weight"}, %arg5: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_0_input_layernorm_weight"}, %arg6: tensor<1x1xi32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1xi32>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg7: tensor<128256x1536xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128256x1536xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_embed_tokens__forward_method___self___weight"}, %arg8: tensor<512x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_0_self_attn_qkv_proj_k_weight"}, %arg9: tensor<1536x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1536x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_0_self_attn_qkv_proj_q_weight"}, %arg10: tensor<7x4x32x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<7x4x32x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "sub__import_vllm_dot_forward_context____forward_context_no_compile_layers__model_layers_0_self_attn_attn___kv_cache_0"}, %arg11: tensor<7x4x32x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<7x4x32x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "sub__import_vllm_dot_forward_context____forward_context_no_compile_layers__model_layers_0_self_attn_attn___kv_cache_1"}, %arg12: tensor<64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<64xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_1_self_attn_rotary_emb_inv_freq"}, %arg13: tensor<512x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_1_self_attn_qkv_proj_v_weight"}, %arg14: tensor<3072x1536xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x1536xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_0_self_attn_o_proj_weight"}, %arg15: tensor<3072x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x4096xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_0_mlp_down_proj_weight"}, %arg16: tensor<4096x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4096x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "self___model_model_layers_0_mlp_gate_up_proj_weights_1"}, %arg17: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_0_post_attention_layernorm_weight"}, %arg18: tensor<4096x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4096x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "self___model_model_layers_0_mlp_gate_up_proj_weights_0"}, %arg19: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_1_input_layernorm_weight"}, %arg20: tensor<512x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_1_self_attn_qkv_proj_k_weight"}, %arg21: tensor<1536x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1536x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_1_self_attn_qkv_proj_q_weight"}, %arg22: tensor<7x4x32x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<7x4x32x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "sub__import_vllm_dot_forward_context____forward_context_no_compile_layers__model_layers_1_self_attn_attn___kv_cache_0"}, %arg23: tensor<7x4x32x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<7x4x32x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "sub__import_vllm_dot_forward_context____forward_context_no_compile_layers__model_layers_1_self_attn_attn___kv_cache_1"}, %arg24: tensor<3072x1536xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x1536xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_1_self_attn_o_proj_weight"}, %arg25: tensor<3072x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x4096xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_1_mlp_down_proj_weight"}, %arg26: tensor<4096x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4096x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "self___model_model_layers_1_mlp_gate_up_proj_weights_1"}, %arg27: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_1_post_attention_layernorm_weight"}, %arg28: tensor<4096x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4096x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "self___model_model_layers_1_mlp_gate_up_proj_weights_0"}, %arg29: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_model_layers_2_input_layernorm_weight"}) -> (tensor<7x4x32x128xbf16>, tensor<7x4x32x128xbf16>, tensor<7x4x32x128xbf16>, tensor<7x4x32x128xbf16>, tensor<1x1x3072xbf16>) {
        %0 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
        %1 = "ttir.constant"() <{value = dense<3.25520843E-4> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
        %2 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x1x3072xf32>
        %5 = "ttir.typecast"(%arg6) <{conservative_folding = false}> : (tensor<1x1xi32>) -> tensor<1x1xi64>
        %6 = "ttir.reshape"(%5) <{shape = [1 : i32]}> : (tensor<1x1xi64>) -> tensor<1xi64>
        %7 = "ttir.typecast"(%6) <{conservative_folding = false}> : (tensor<1xi64>) -> tensor<1xui32>
        %8 = "ttir.permute"(%arg7) <{permutation = array<i64: 0, 1>}> : (tensor<128256x1536xbf16>) -> tensor<128256x1536xbf16>
        %9 = "ttir.reshape"(%8) <{shape = [128256 : i32, 1536 : i32]}> : (tensor<128256x1536xbf16>) -> tensor<128256x1536xbf16>
        %10 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xui32>) -> tensor<1x1xui32>
        %11 = "ttir.embedding"(%10, %9) : (tensor<1x1xui32>, tensor<128256x1536xbf16>) -> tensor<1x1x1536xbf16>
        %12 = "ttir.reshape"(%11) <{shape = [1 : i32, 1536 : i32]}> : (tensor<1x1x1536xbf16>) -> tensor<1x1536xbf16>
        %13 = "ttir.permute"(%12) <{permutation = array<i64: 0, 1>}> : (tensor<1x1536xbf16>) -> tensor<1x1536xbf16>
        %14 = "ttir.reshape"(%13) <{shape = [1 : i32, 1 : i32, 1536 : i32]}> : (tensor<1x1536xbf16>) -> tensor<1x1x1536xbf16>
        %15 = "ttir.all_gather"(%14) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x1536xbf16>) -> tensor<1x1x3072xbf16>
        %16 = "ttir.rms_norm"(%15, %arg5) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 3072>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x3072xbf16>, tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
        %17 = "ttir.reshape"(%16) <{shape = [1 : i32, 3072 : i32]}> : (tensor<1x1x3072xbf16>) -> tensor<1x3072xbf16>
        %18 = "ttir.permute"(%arg9) <{permutation = array<i64: 1, 0>}> : (tensor<1536x3072xbf16>) -> tensor<3072x1536xbf16>
        %19 = "ttir.dot_general"(%17, %18) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x1536xbf16>) -> tensor<1x1536xbf16>
        %20 = "ttir.permute"(%arg8) <{permutation = array<i64: 1, 0>}> : (tensor<512x3072xbf16>) -> tensor<3072x512xbf16>
        %21 = "ttir.dot_general"(%17, %20) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x512xbf16>) -> tensor<1x512xbf16>
        %22 = "ttir.permute"(%arg4) <{permutation = array<i64: 1, 0>}> : (tensor<512x3072xbf16>) -> tensor<3072x512xbf16>
        %23 = "ttir.dot_general"(%17, %22) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x512xbf16>) -> tensor<1x512xbf16>
        %24 = "ttir.reshape"(%21) <{shape = [1 : i32, 4 : i32, 128 : i32]}> : (tensor<1x512xbf16>) -> tensor<1x4x128xbf16>
        %25 = "ttir.slice_static"(%24) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4x128xbf16>) -> tensor<1x4x64xbf16>
        %26 = "ttir.reshape"(%arg3) <{shape = [1 : i32]}> : (tensor<1x1xi32>) -> tensor<1xi32>
        %27 = "ttir.typecast"(%26) <{conservative_folding = false}> : (tensor<1xi32>) -> tensor<1xf32>
        %28 = "ttir.reshape"(%27) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>) -> tensor<1x1xf32>
        %29 = "ttir.broadcast"(%28) <{broadcast_dimensions = array<i64: 1, 64>}> : (tensor<1x1xf32>) -> tensor<1x64xf32>
        %30 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 64 : i32]}> : (tensor<64xf32>) -> tensor<1x64xf32>
        %31 = "ttir.multiply"(%29, %30) : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
        %32 = "ttir.cos"(%31) : (tensor<1x64xf32>) -> tensor<1x64xf32>
        %33 = "ttir.typecast"(%32) <{conservative_folding = false}> : (tensor<1x64xf32>) -> tensor<1x64xbf16>
        %34 = "ttir.reshape"(%33) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x64xbf16>
        %35 = "ttir.broadcast"(%34) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<1x1x64xbf16>) -> tensor<1x4x64xbf16>
        %36 = "ttir.multiply"(%25, %35) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %37 = "ttir.slice_static"(%24) <{begins = [0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 4 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4x128xbf16>) -> tensor<1x4x64xbf16>
        %38 = "ttir.sin"(%31) : (tensor<1x64xf32>) -> tensor<1x64xf32>
        %39 = "ttir.typecast"(%38) <{conservative_folding = false}> : (tensor<1x64xf32>) -> tensor<1x64xbf16>
        %40 = "ttir.reshape"(%39) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x64xbf16>
        %41 = "ttir.broadcast"(%40) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<1x1x64xbf16>) -> tensor<1x4x64xbf16>
        %42 = "ttir.multiply"(%37, %41) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %43 = "ttir.subtract"(%36, %42) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %44 = "ttir.multiply"(%37, %35) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %45 = "ttir.multiply"(%25, %41) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %46 = "ttir.add"(%44, %45) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %47 = "ttir.concat"(%43, %46) <{dim = 2 : si32}> : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x128xbf16>
        %48 = "ttir.reshape"(%47) <{shape = [1 : i32, 1 : i32, 4 : i32, 128 : i32]}> : (tensor<1x4x128xbf16>) -> tensor<1x1x4x128xbf16>
        %49 = "ttir.paged_update_cache"(%arg10, %48, %arg1, %arg0) <{share_cache = false}> : (tensor<7x4x32x128xbf16>, tensor<1x1x4x128xbf16>, tensor<1xi32>, tensor<1x1xi32>) -> tensor<7x4x32x128xbf16>
        %50 = "ttir.reshape"(%23) <{shape = [1 : i32, 1 : i32, 4 : i32, 128 : i32]}> : (tensor<1x512xbf16>) -> tensor<1x1x4x128xbf16>
        %51 = "ttir.paged_update_cache"(%arg11, %50, %arg1, %arg0) <{share_cache = false}> : (tensor<7x4x32x128xbf16>, tensor<1x1x4x128xbf16>, tensor<1xi32>, tensor<1x1xi32>) -> tensor<7x4x32x128xbf16>
        %52 = "ttir.reshape"(%arg19) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
        %53 = "ttir.reshape"(%arg17) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
        %54 = "ttir.reshape"(%19) <{shape = [1 : i32, 12 : i32, 128 : i32]}> : (tensor<1x1536xbf16>) -> tensor<1x12x128xbf16>
        %55 = "ttir.slice_static"(%54) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 12 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x12x128xbf16>) -> tensor<1x12x64xbf16>
        %56 = "ttir.broadcast"(%34) <{broadcast_dimensions = array<i64: 1, 12, 1>}> : (tensor<1x1x64xbf16>) -> tensor<1x12x64xbf16>
        %57 = "ttir.multiply"(%55, %56) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %58 = "ttir.slice_static"(%54) <{begins = [0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 12 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x12x128xbf16>) -> tensor<1x12x64xbf16>
        %59 = "ttir.broadcast"(%40) <{broadcast_dimensions = array<i64: 1, 12, 1>}> : (tensor<1x1x64xbf16>) -> tensor<1x12x64xbf16>
        %60 = "ttir.multiply"(%58, %59) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %61 = "ttir.subtract"(%57, %60) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %62 = "ttir.multiply"(%58, %56) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %63 = "ttir.multiply"(%55, %59) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %64 = "ttir.add"(%62, %63) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %65 = "ttir.concat"(%61, %64) <{dim = 2 : si32}> : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x128xbf16>
        %66 = "ttir.reshape"(%65) <{shape = [1 : i32, 1 : i32, 12 : i32, 128 : i32]}> : (tensor<1x12x128xbf16>) -> tensor<1x1x12x128xbf16>
        %67 = ttir.empty() : tensor<1x1x12x128xbf16>
        %68 = "ttir.paged_scaled_dot_product_attention_decode"(%66, %49, %51, %arg0, %67, %arg1) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 0>, scale = 0.0883883461 : f32}> : (tensor<1x1x12x128xbf16>, tensor<7x4x32x128xbf16>, tensor<7x4x32x128xbf16>, tensor<1x1xi32>, tensor<1x1x12x128xbf16>, tensor<1xi32>) -> tensor<1x1x12x128xbf16>
        %69 = "ttir.reshape"(%68) <{shape = [1 : i32, 1536 : i32]}> : (tensor<1x1x12x128xbf16>) -> tensor<1x1536xbf16>
        %70 = "ttir.permute"(%arg14) <{permutation = array<i64: 1, 0>}> : (tensor<3072x1536xbf16>) -> tensor<1536x3072xbf16>
        %71 = "ttir.dot_general"(%69, %70) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x1536xbf16>, tensor<1536x3072xbf16>) -> tensor<1x3072xbf16>
        %72 = "ttir.all_reduce"(%71) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x3072xbf16>) -> tensor<1x3072xbf16>
        %73 = "ttir.reshape"(%72) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<1x3072xbf16>) -> tensor<1x1x3072xbf16>
        %74 = "ttir.typecast"(%73) <{conservative_folding = false}> : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
        %75 = "ttir.typecast"(%15) <{conservative_folding = false}> : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
        %76 = "ttir.add"(%74, %75) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %77 = "ttir.pow"(%76, %4) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %78 = "ttir.sum"(%77) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1xf32>
        %79 = "ttir.multiply"(%78, %1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
        %80 = "ttir.reshape"(%79) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %81 = "ttir.add"(%80, %0) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %82 = "ttir.rsqrt"(%81) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %83 = "ttir.reshape"(%82) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
        %84 = "ttir.reshape"(%83) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %85 = "ttir.broadcast"(%84) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x1x3072xf32>
        %86 = "ttir.multiply"(%76, %85) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %87 = "ttir.typecast"(%86) <{conservative_folding = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xbf16>
        %88 = "ttir.multiply"(%53, %87) : (tensor<1x1x3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
        %89 = "ttir.reshape"(%88) <{shape = [1 : i32, 3072 : i32]}> : (tensor<1x1x3072xbf16>) -> tensor<1x3072xbf16>
        %90 = "ttir.permute"(%arg18) <{permutation = array<i64: 1, 0>}> : (tensor<4096x3072xbf16>) -> tensor<3072x4096xbf16>
        %91 = "ttir.dot_general"(%89, %90) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x4096xbf16>) -> tensor<1x4096xbf16>
        %92 = "ttir.reshape"(%91) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x4096xbf16>) -> tensor<1x1x4096xbf16>
        %93 = "ttir.permute"(%arg16) <{permutation = array<i64: 1, 0>}> : (tensor<4096x3072xbf16>) -> tensor<3072x4096xbf16>
        %94 = "ttir.dot_general"(%89, %93) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x4096xbf16>) -> tensor<1x4096xbf16>
        %95 = "ttir.reshape"(%94) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x4096xbf16>) -> tensor<1x1x4096xbf16>
        %96 = "ttir.typecast"(%92) <{conservative_folding = false}> : (tensor<1x1x4096xbf16>) -> tensor<1x1x4096xf32>
        %97 = "ttir.sigmoid"(%96) : (tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
        %98 = "ttir.multiply"(%96, %97) : (tensor<1x1x4096xf32>, tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
        %99 = "ttir.typecast"(%98) <{conservative_folding = false}> : (tensor<1x1x4096xf32>) -> tensor<1x1x4096xbf16>
        %100 = "ttir.multiply"(%99, %95) : (tensor<1x1x4096xbf16>, tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
        %101 = "ttir.reshape"(%100) <{shape = [1 : i32, 4096 : i32]}> : (tensor<1x1x4096xbf16>) -> tensor<1x4096xbf16>
        %102 = "ttir.permute"(%arg15) <{permutation = array<i64: 1, 0>}> : (tensor<3072x4096xbf16>) -> tensor<4096x3072xbf16>
        %103 = "ttir.dot_general"(%101, %102) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x4096xbf16>, tensor<4096x3072xbf16>) -> tensor<1x3072xbf16>
        %104 = "ttir.all_reduce"(%103) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x3072xbf16>) -> tensor<1x3072xbf16>
        %105 = "ttir.reshape"(%104) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<1x3072xbf16>) -> tensor<1x1x3072xbf16>
        %106 = "ttir.typecast"(%105) <{conservative_folding = false}> : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
        %107 = "ttir.typecast"(%76) <{conservative_folding = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xbf16>
        %108 = "ttir.typecast"(%107) <{conservative_folding = false}> : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
        %109 = "ttir.add"(%106, %108) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %110 = "ttir.pow"(%109, %4) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %111 = "ttir.sum"(%110) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1xf32>
        %112 = "ttir.multiply"(%111, %1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
        %113 = "ttir.reshape"(%112) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %114 = "ttir.add"(%113, %0) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %115 = "ttir.rsqrt"(%114) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %116 = "ttir.reshape"(%115) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
        %117 = "ttir.reshape"(%116) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %118 = "ttir.broadcast"(%117) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x1x3072xf32>
        %119 = "ttir.multiply"(%109, %118) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %120 = "ttir.typecast"(%119) <{conservative_folding = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xbf16>
        %121 = "ttir.multiply"(%52, %120) : (tensor<1x1x3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
        %122 = "ttir.reshape"(%121) <{shape = [1 : i32, 3072 : i32]}> : (tensor<1x1x3072xbf16>) -> tensor<1x3072xbf16>
        %123 = "ttir.permute"(%arg21) <{permutation = array<i64: 1, 0>}> : (tensor<1536x3072xbf16>) -> tensor<3072x1536xbf16>
        %124 = "ttir.dot_general"(%122, %123) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x1536xbf16>) -> tensor<1x1536xbf16>
        %125 = "ttir.permute"(%arg20) <{permutation = array<i64: 1, 0>}> : (tensor<512x3072xbf16>) -> tensor<3072x512xbf16>
        %126 = "ttir.dot_general"(%122, %125) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x512xbf16>) -> tensor<1x512xbf16>
        %127 = "ttir.permute"(%arg13) <{permutation = array<i64: 1, 0>}> : (tensor<512x3072xbf16>) -> tensor<3072x512xbf16>
        %128 = "ttir.dot_general"(%122, %127) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x512xbf16>) -> tensor<1x512xbf16>
        %129 = "ttir.reshape"(%126) <{shape = [1 : i32, 4 : i32, 128 : i32]}> : (tensor<1x512xbf16>) -> tensor<1x4x128xbf16>
        %130 = "ttir.slice_static"(%129) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4x128xbf16>) -> tensor<1x4x64xbf16>
        %131 = "ttir.reshape"(%arg12) <{shape = [1 : i32, 64 : i32]}> : (tensor<64xf32>) -> tensor<1x64xf32>
        %132 = "ttir.multiply"(%29, %131) : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
        %133 = "ttir.cos"(%132) : (tensor<1x64xf32>) -> tensor<1x64xf32>
        %134 = "ttir.typecast"(%133) <{conservative_folding = false}> : (tensor<1x64xf32>) -> tensor<1x64xbf16>
        %135 = "ttir.reshape"(%134) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x64xbf16>
        %136 = "ttir.broadcast"(%135) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<1x1x64xbf16>) -> tensor<1x4x64xbf16>
        %137 = "ttir.multiply"(%130, %136) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %138 = "ttir.slice_static"(%129) <{begins = [0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 4 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4x128xbf16>) -> tensor<1x4x64xbf16>
        %139 = "ttir.sin"(%132) : (tensor<1x64xf32>) -> tensor<1x64xf32>
        %140 = "ttir.typecast"(%139) <{conservative_folding = false}> : (tensor<1x64xf32>) -> tensor<1x64xbf16>
        %141 = "ttir.reshape"(%140) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x64xbf16>
        %142 = "ttir.broadcast"(%141) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<1x1x64xbf16>) -> tensor<1x4x64xbf16>
        %143 = "ttir.multiply"(%138, %142) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %144 = "ttir.subtract"(%137, %143) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %145 = "ttir.multiply"(%138, %136) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %146 = "ttir.multiply"(%130, %142) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %147 = "ttir.add"(%145, %146) : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x64xbf16>
        %148 = "ttir.concat"(%144, %147) <{dim = 2 : si32}> : (tensor<1x4x64xbf16>, tensor<1x4x64xbf16>) -> tensor<1x4x128xbf16>
        %149 = "ttir.reshape"(%148) <{shape = [1 : i32, 1 : i32, 4 : i32, 128 : i32]}> : (tensor<1x4x128xbf16>) -> tensor<1x1x4x128xbf16>
        %150 = "ttir.paged_update_cache"(%arg22, %149, %arg1, %arg0) <{share_cache = false}> : (tensor<7x4x32x128xbf16>, tensor<1x1x4x128xbf16>, tensor<1xi32>, tensor<1x1xi32>) -> tensor<7x4x32x128xbf16>
        %151 = "ttir.reshape"(%128) <{shape = [1 : i32, 1 : i32, 4 : i32, 128 : i32]}> : (tensor<1x512xbf16>) -> tensor<1x1x4x128xbf16>
        %152 = "ttir.paged_update_cache"(%arg23, %151, %arg1, %arg0) <{share_cache = false}> : (tensor<7x4x32x128xbf16>, tensor<1x1x4x128xbf16>, tensor<1xi32>, tensor<1x1xi32>) -> tensor<7x4x32x128xbf16>
        %153 = "ttir.reshape"(%arg29) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
        %154 = "ttir.reshape"(%arg27) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
        %155 = "ttir.reshape"(%124) <{shape = [1 : i32, 12 : i32, 128 : i32]}> : (tensor<1x1536xbf16>) -> tensor<1x12x128xbf16>
        %156 = "ttir.slice_static"(%155) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 12 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x12x128xbf16>) -> tensor<1x12x64xbf16>
        %157 = "ttir.broadcast"(%135) <{broadcast_dimensions = array<i64: 1, 12, 1>}> : (tensor<1x1x64xbf16>) -> tensor<1x12x64xbf16>
        %158 = "ttir.multiply"(%156, %157) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %159 = "ttir.slice_static"(%155) <{begins = [0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 12 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x12x128xbf16>) -> tensor<1x12x64xbf16>
        %160 = "ttir.broadcast"(%141) <{broadcast_dimensions = array<i64: 1, 12, 1>}> : (tensor<1x1x64xbf16>) -> tensor<1x12x64xbf16>
        %161 = "ttir.multiply"(%159, %160) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %162 = "ttir.subtract"(%158, %161) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %163 = "ttir.multiply"(%159, %157) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %164 = "ttir.multiply"(%156, %160) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %165 = "ttir.add"(%163, %164) : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x64xbf16>
        %166 = "ttir.concat"(%162, %165) <{dim = 2 : si32}> : (tensor<1x12x64xbf16>, tensor<1x12x64xbf16>) -> tensor<1x12x128xbf16>
        %167 = "ttir.reshape"(%166) <{shape = [1 : i32, 1 : i32, 12 : i32, 128 : i32]}> : (tensor<1x12x128xbf16>) -> tensor<1x1x12x128xbf16>
        %168 = ttir.empty() : tensor<1x1x12x128xbf16>
        %169 = "ttir.paged_scaled_dot_product_attention_decode"(%167, %150, %152, %arg0, %168, %arg1) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 0>, scale = 0.0883883461 : f32}> : (tensor<1x1x12x128xbf16>, tensor<7x4x32x128xbf16>, tensor<7x4x32x128xbf16>, tensor<1x1xi32>, tensor<1x1x12x128xbf16>, tensor<1xi32>) -> tensor<1x1x12x128xbf16>
        %170 = "ttir.reshape"(%169) <{shape = [1 : i32, 1536 : i32]}> : (tensor<1x1x12x128xbf16>) -> tensor<1x1536xbf16>
        %171 = "ttir.permute"(%arg24) <{permutation = array<i64: 1, 0>}> : (tensor<3072x1536xbf16>) -> tensor<1536x3072xbf16>
        %172 = "ttir.dot_general"(%170, %171) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x1536xbf16>, tensor<1536x3072xbf16>) -> tensor<1x3072xbf16>
        %173 = "ttir.all_reduce"(%172) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x3072xbf16>) -> tensor<1x3072xbf16>
        %174 = "ttir.reshape"(%173) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<1x3072xbf16>) -> tensor<1x1x3072xbf16>
        %175 = "ttir.typecast"(%174) <{conservative_folding = false}> : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
        %176 = "ttir.typecast"(%109) <{conservative_folding = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xbf16>
        %177 = "ttir.typecast"(%176) <{conservative_folding = false}> : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
        %178 = "ttir.add"(%175, %177) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %179 = "ttir.pow"(%178, %4) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %180 = "ttir.sum"(%179) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1xf32>
        %181 = "ttir.multiply"(%180, %1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
        %182 = "ttir.reshape"(%181) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %183 = "ttir.add"(%182, %0) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %184 = "ttir.rsqrt"(%183) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %185 = "ttir.reshape"(%184) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
        %186 = "ttir.reshape"(%185) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %187 = "ttir.broadcast"(%186) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x1x3072xf32>
        %188 = "ttir.multiply"(%178, %187) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %189 = "ttir.typecast"(%188) <{conservative_folding = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xbf16>
        %190 = "ttir.multiply"(%154, %189) : (tensor<1x1x3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
        %191 = "ttir.reshape"(%190) <{shape = [1 : i32, 3072 : i32]}> : (tensor<1x1x3072xbf16>) -> tensor<1x3072xbf16>
        %192 = "ttir.permute"(%arg28) <{permutation = array<i64: 1, 0>}> : (tensor<4096x3072xbf16>) -> tensor<3072x4096xbf16>
        %193 = "ttir.dot_general"(%191, %192) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x4096xbf16>) -> tensor<1x4096xbf16>
        %194 = "ttir.reshape"(%193) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x4096xbf16>) -> tensor<1x1x4096xbf16>
        %195 = "ttir.permute"(%arg26) <{permutation = array<i64: 1, 0>}> : (tensor<4096x3072xbf16>) -> tensor<3072x4096xbf16>
        %196 = "ttir.dot_general"(%191, %195) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x4096xbf16>) -> tensor<1x4096xbf16>
        %197 = "ttir.reshape"(%196) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x4096xbf16>) -> tensor<1x1x4096xbf16>
        %198 = "ttir.typecast"(%194) <{conservative_folding = false}> : (tensor<1x1x4096xbf16>) -> tensor<1x1x4096xf32>
        %199 = "ttir.sigmoid"(%198) : (tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
        %200 = "ttir.multiply"(%198, %199) : (tensor<1x1x4096xf32>, tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
        %201 = "ttir.typecast"(%200) <{conservative_folding = false}> : (tensor<1x1x4096xf32>) -> tensor<1x1x4096xbf16>
        %202 = "ttir.multiply"(%201, %197) : (tensor<1x1x4096xbf16>, tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
        %203 = "ttir.reshape"(%202) <{shape = [1 : i32, 4096 : i32]}> : (tensor<1x1x4096xbf16>) -> tensor<1x4096xbf16>
        %204 = "ttir.permute"(%arg25) <{permutation = array<i64: 1, 0>}> : (tensor<3072x4096xbf16>) -> tensor<4096x3072xbf16>
        %205 = "ttir.dot_general"(%203, %204) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x4096xbf16>, tensor<4096x3072xbf16>) -> tensor<1x3072xbf16>
        %206 = "ttir.all_reduce"(%205) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x3072xbf16>) -> tensor<1x3072xbf16>
        %207 = "ttir.reshape"(%206) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<1x3072xbf16>) -> tensor<1x1x3072xbf16>
        %208 = "ttir.typecast"(%207) <{conservative_folding = false}> : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
        %209 = "ttir.typecast"(%178) <{conservative_folding = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xbf16>
        %210 = "ttir.typecast"(%209) <{conservative_folding = false}> : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
        %211 = "ttir.add"(%208, %210) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %212 = "ttir.pow"(%211, %4) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %213 = "ttir.sum"(%212) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1xf32>
        %214 = "ttir.multiply"(%213, %1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
        %215 = "ttir.reshape"(%214) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %216 = "ttir.add"(%215, %0) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %217 = "ttir.rsqrt"(%216) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %218 = "ttir.reshape"(%217) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
        %219 = "ttir.reshape"(%218) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %220 = "ttir.broadcast"(%219) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x1x3072xf32>
        %221 = "ttir.multiply"(%211, %220) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %222 = "ttir.typecast"(%221) <{conservative_folding = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xbf16>
        %223 = "ttir.multiply"(%153, %222) : (tensor<1x1x3072xbf16>, tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
        return %49, %51, %150, %152, %223 : tensor<7x4x32x128xbf16>, tensor<7x4x32x128xbf16>, tensor<7x4x32x128xbf16>, tensor<7x4x32x128xbf16>, tensor<1x1x3072xbf16>
      }
    }
  }
}
