func.func @llama_3_2_1b_1_layer(%arg0: tensor<1xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_1"} loc("p0.3"), %arg1: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_rotary_emb_inv_freq"} loc("p1.13"), %arg2: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_k_proj_weight"} loc("p2.31"), %arg3: tensor<32x1xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p3.39"), %arg4: tensor<128256x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_embed_tokens_weight"} loc("p4.44"), %arg5: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___input_layernorm_weight"} loc("p5.80"), %arg6: tensor<32x8x128x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_2"} loc("p6.120"), %arg7: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_v_proj_weight"} loc("p7.128"), %arg8: tensor<32x8x128x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_3"} loc("p8.148"), %arg9: tensor<128256x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___lm_head_weight"} loc("p9.156"), %arg10: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___mlp_down_proj_weight"} loc("p10.165"), %arg11: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___mlp_up_proj_weight"} loc("p11.170"), %arg12: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_o_proj_weight"} loc("p12.179"), %arg13: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_q_proj_weight"} loc("p13.229"), %arg14: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___post_attention_layernorm_weight"} loc("p14.319"), %arg15: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___mlp_gate_proj_weight"} loc("p15.328"), %arg16: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_norm_weight"} loc("p16.374")) -> (tensor<32x8x128x64xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<32x8x128x64xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<32x1x128256xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = "ttir.constant"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F00000000000000"> : tensor<1x128xi64>}> : () -> tensor<1x128xi64>
    %1 = "ttir.constant"() <{value = dense<-3.389530e+38> : tensor<1x128xbf16>}> : () -> tensor<1x128xbf16>
    %2 = "ttir.constant"() <{value = dense<1.250000e-01> : tensor<32x32x1x128xbf16>}> : () -> tensor<32x32x1x128xbf16>
    %3 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<32x1x1xf32>}> : () -> tensor<32x1x1xf32>
    %4 = "ttir.constant"() <{value = dense<4.8828125E-4> : tensor<32x1xf32>}> : () -> tensor<32x1xf32>
    %5 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x1x2048xf32>}> : () -> tensor<32x1x2048xf32>
    %6 = "ttir.constant"() <{value = dense<128> : tensor<1xi64>}> : () -> tensor<1xi64>
    %7 = "ttir.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %8 = "ttir.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %9 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %10 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xi64>) -> tensor<1x1x1xi64>
    %11 = "ttir.reshape"(%10) <{shape = [1 : i32]}> : (tensor<1x1x1xi64>) -> tensor<1xi64>
    %12 = "ttir.lt"(%11, %7) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %13 = "ttir.add"(%11, %6) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %14 = "ttir.where"(%12, %13, %11) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %15 = "ttir.reshape"(%14) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi64>) -> tensor<1x1xi64>
    %16 = "ttir.reshape"(%arg5) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %17 = "ttir.reshape"(%16) <{shape = [2048 : i32]}> : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %18 = "ttir.reshape"(%17) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %19 = "ttir.broadcast"(%18) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %20 = "ttir.reshape"(%arg4) <{shape = [1 : i32, 128256 : i32, 2048 : i32]}> : (tensor<128256x2048xbf16>) -> tensor<1x128256x2048xbf16>
    %21 = "ttir.reshape"(%20) <{shape = [128256 : i32, 2048 : i32]}> : (tensor<1x128256x2048xbf16>) -> tensor<128256x2048xbf16>
    %22 = "ttir.reshape"(%arg3) <{shape = [1 : i32, 32 : i32, 1 : i32]}> : (tensor<32x1xi64>) -> tensor<1x32x1xi64>
    %23 = "ttir.reshape"(%22) <{shape = [32 : i32]}> : (tensor<1x32x1xi64>) -> tensor<32xi64>
    %24 = "ttir.typecast"(%23) <{conservative_folding = false}> : (tensor<32xi64>) -> tensor<32xui32>
    %25 = "ttir.gather"(%21, %24) <{collapsed_slice_dims = array<i64: 0>, index_vector_dim = 1 : si64, indices_are_sorted = false, offset_dims = array<i64: 1>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 2048>, start_index_map = array<i64: 0>, start_indices_batching_dims = array<i64>}> : (tensor<128256x2048xbf16>, tensor<32xui32>) -> tensor<32x2048xbf16>
    %26 = "ttir.reshape"(%25) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
    %27 = "ttir.typecast"(%26) <{conservative_folding = false}> : (tensor<32x1x2048xbf16>) -> tensor<32x1x2048xf32>
    %28 = "ttir.pow"(%27, %5) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
    %29 = "ttir.sum"(%28) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1xf32>
    %30 = "ttir.multiply"(%29, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %31 = "ttir.reshape"(%30) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %32 = "ttir.add"(%31, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %33 = "ttir.rsqrt"(%32) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %34 = "ttir.reshape"(%33) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
    %35 = "ttir.reshape"(%34) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %36 = "ttir.broadcast"(%35) <{broadcast_dimensions = array<i64: 1, 1, 2048>}> : (tensor<32x1x1xf32>) -> tensor<32x1x2048xf32>
    %37 = "ttir.multiply"(%27, %36) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
    %38 = "ttir.typecast"(%37) <{conservative_folding = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1x2048xbf16>
    %39 = "ttir.multiply"(%19, %38) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %40 = "ttir.reshape"(%39) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
    %41 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %42 = "ttir.reshape"(%41) <{shape = [512 : i32, 2048 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %43 = "ttir.permute"(%42) <{permutation = array<i64: 1, 0>}> : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %44 = "ttir.dot_general"(%40, %43) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x512xbf16>) -> tensor<32x512xbf16>
    %45 = "ttir.reshape"(%44) <{shape = [32 : i32, 8 : i32, 1 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<32x8x1x64xbf16>
    %46 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<32xbf16>) -> tensor<1x1x32xbf16>
    %47 = "ttir.reshape"(%46) <{shape = [1 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x32xbf16>) -> tensor<1x32x1xbf16>
    %48 = "ttir.typecast"(%47) <{conservative_folding = false}> : (tensor<1x32x1xbf16>) -> tensor<1x32x1xf32>
    %49 = "ttir.typecast"(%10) <{conservative_folding = false}> : (tensor<1x1x1xi64>) -> tensor<1x1x1xf32>
    %50 = "ttir.dot_general"(%48, %49) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<1x32x1xf32>, tensor<1x1x1xf32>) -> tensor<1x32x1xf32>
    %51 = "ttir.reshape"(%50) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x32x1xf32>) -> tensor<1x1x32xf32>
    %52 = "ttir.concat"(%51, %51) <{dim = 2 : si32}> : (tensor<1x1x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x64xf32>
    %53 = "ttir.cos"(%52) : (tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
    %54 = "ttir.typecast"(%53) <{conservative_folding = false}> : (tensor<1x1x64xf32>) -> tensor<1x1x64xbf16>
    %55 = "ttir.reshape"(%54) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x1x64xbf16>) -> tensor<1x64xbf16>
    %56 = "ttir.reshape"(%55) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x1x64xbf16>
    %57 = "ttir.broadcast"(%56) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %58 = "ttir.multiply"(%45, %57) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %59 = "ttir.slice_static"(%45) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %60 = "ttir.neg"(%59) : (tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %61 = "ttir.slice_static"(%45) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %62 = "ttir.concat"(%60, %61) <{dim = 3 : si32}> : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x64xbf16>
    %63 = "ttir.sin"(%52) : (tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
    %64 = "ttir.typecast"(%63) <{conservative_folding = false}> : (tensor<1x1x64xf32>) -> tensor<1x1x64xbf16>
    %65 = "ttir.reshape"(%64) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x1x64xbf16>) -> tensor<1x64xbf16>
    %66 = "ttir.reshape"(%65) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x1x64xbf16>
    %67 = "ttir.broadcast"(%66) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %68 = "ttir.multiply"(%62, %67) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %69 = "ttir.add"(%58, %68) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %70 = "ttir.permute"(%69) <{permutation = array<i64: 2, 1, 0, 3>}> : (tensor<32x8x1x64xbf16>) -> tensor<1x8x32x64xbf16>
    %71 = "ttir.update_cache"(%arg6, %70, %arg0) <{batch_offset = 0 : i32}> : (tensor<32x8x128x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1xi64>) -> tensor<32x8x128x64xbf16>
    %72 = "ttir.reshape"(%arg7) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %73 = "ttir.reshape"(%72) <{shape = [512 : i32, 2048 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %74 = "ttir.permute"(%73) <{permutation = array<i64: 1, 0>}> : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %75 = "ttir.dot_general"(%40, %74) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x512xbf16>) -> tensor<32x512xbf16>
    %76 = "ttir.reshape"(%75) <{shape = [32 : i32, 8 : i32, 1 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<32x8x1x64xbf16>
    %77 = "ttir.permute"(%76) <{permutation = array<i64: 2, 1, 0, 3>}> : (tensor<32x8x1x64xbf16>) -> tensor<1x8x32x64xbf16>
    %78 = "ttir.update_cache"(%arg8, %77, %arg0) <{batch_offset = 0 : i32}> : (tensor<32x8x128x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1xi64>) -> tensor<32x8x128x64xbf16>
    %79 = "ttir.reshape"(%arg16) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %80 = "ttir.reshape"(%79) <{shape = [2048 : i32]}> : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %81 = "ttir.reshape"(%80) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %82 = "ttir.broadcast"(%81) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %83 = "ttir.reshape"(%arg13) <{shape = [1 : i32, 2048 : i32, 2048 : i32]}> : (tensor<2048x2048xbf16>) -> tensor<1x2048x2048xbf16>
    %84 = "ttir.reshape"(%83) <{shape = [2048 : i32, 2048 : i32]}> : (tensor<1x2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %85 = "ttir.permute"(%84) <{permutation = array<i64: 1, 0>}> : (tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %86 = "ttir.dot_general"(%40, %85) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<32x2048xbf16>
    %87 = "ttir.reshape"(%86) <{shape = [32 : i32, 32 : i32, 1 : i32, 64 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x32x1x64xbf16>
    %88 = "ttir.reshape"(%55) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x1x64xbf16>
    %89 = "ttir.broadcast"(%88) <{broadcast_dimensions = array<i64: 32, 32, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x32x1x64xbf16>
    %90 = "ttir.multiply"(%87, %89) : (tensor<32x32x1x64xbf16>, tensor<32x32x1x64xbf16>) -> tensor<32x32x1x64xbf16>
    %91 = "ttir.slice_static"(%87) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [32 : i32, 32 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1x64xbf16>) -> tensor<32x32x1x32xbf16>
    %92 = "ttir.neg"(%91) : (tensor<32x32x1x32xbf16>) -> tensor<32x32x1x32xbf16>
    %93 = "ttir.slice_static"(%87) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 32 : i32, 1 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1x64xbf16>) -> tensor<32x32x1x32xbf16>
    %94 = "ttir.concat"(%92, %93) <{dim = 3 : si32}> : (tensor<32x32x1x32xbf16>, tensor<32x32x1x32xbf16>) -> tensor<32x32x1x64xbf16>
    %95 = "ttir.reshape"(%65) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x1x64xbf16>
    %96 = "ttir.broadcast"(%95) <{broadcast_dimensions = array<i64: 32, 32, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x32x1x64xbf16>
    %97 = "ttir.multiply"(%94, %96) : (tensor<32x32x1x64xbf16>, tensor<32x32x1x64xbf16>) -> tensor<32x32x1x64xbf16>
    %98 = "ttir.add"(%90, %97) : (tensor<32x32x1x64xbf16>, tensor<32x32x1x64xbf16>) -> tensor<32x32x1x64xbf16>
    %99 = "ttir.reshape"(%71) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x128x64xbf16>) -> tensor<32x8x1x128x64xbf16>
    %100 = "ttir.broadcast"(%99) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x128x64xbf16>) -> tensor<32x8x4x128x64xbf16>
    %101 = "ttir.reshape"(%100) <{shape = [32 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x4x128x64xbf16>) -> tensor<32x32x128x64xbf16>
    %102 = "ttir.permute"(%101) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<32x32x128x64xbf16>) -> tensor<32x32x64x128xbf16>
    %103 = "ttir.dot_general"(%98, %102) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<32x32x1x64xbf16>, tensor<32x32x64x128xbf16>) -> tensor<32x32x1x128xbf16>
    %104 = "ttir.multiply"(%103, %2) : (tensor<32x32x1x128xbf16>, tensor<32x32x1x128xbf16>) -> tensor<32x32x1x128xbf16>
    %105 = "ttir.reshape"(%11) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi64>) -> tensor<1x1xi64>
    %106 = "ttir.broadcast"(%105) <{broadcast_dimensions = array<i64: 1, 128>}> : (tensor<1x1xi64>) -> tensor<1x128xi64>
    %107 = "ttir.gt"(%0, %106) : (tensor<1x128xi64>, tensor<1x128xi64>) -> tensor<1x128xi1>
    %108 = "ttir.typecast"(%107) <{conservative_folding = false}> : (tensor<1x128xi1>) -> tensor<1x128xbf16>
    %109 = "ttir.multiply"(%108, %1) : (tensor<1x128xbf16>, tensor<1x128xbf16>) -> tensor<1x128xbf16>
    %110 = "ttir.reshape"(%109) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xbf16>) -> tensor<1x1x128xbf16>
    %111 = "ttir.reshape"(%110) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    %112 = "ttir.broadcast"(%111) <{broadcast_dimensions = array<i64: 32, 1, 1, 1>}> : (tensor<1x1x1x128xbf16>) -> tensor<32x1x1x128xbf16>
    %113 = "ttir.reshape"(%112) <{shape = [32 : i32, 1 : i32, 128 : i32]}> : (tensor<32x1x1x128xbf16>) -> tensor<32x1x128xbf16>
    %114 = "ttir.reshape"(%113) <{shape = [32 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<32x1x128xbf16>) -> tensor<32x1x1x128xbf16>
    %115 = "ttir.broadcast"(%114) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<32x1x1x128xbf16>) -> tensor<32x32x1x128xbf16>
    %116 = "ttir.add"(%104, %115) : (tensor<32x32x1x128xbf16>, tensor<32x32x1x128xbf16>) -> tensor<32x32x1x128xbf16>
    %117 = "ttir.typecast"(%116) <{conservative_folding = false}> : (tensor<32x32x1x128xbf16>) -> tensor<32x32x1x128xf32>
    %118 = "ttir.max"(%117) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1xf32>
    %119 = "ttir.reshape"(%118) <{shape = [32 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1xf32>) -> tensor<32x32x1x1xf32>
    %120 = "ttir.broadcast"(%119) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x128xf32>
    %121 = "ttir.subtract"(%117, %120) : (tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %122 = "ttir.exp"(%121) : (tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %123 = "ttir.sum"(%122) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1xf32>
    %124 = "ttir.reshape"(%123) <{shape = [32 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1xf32>) -> tensor<32x32x1x1xf32>
    %125 = "ttir.broadcast"(%124) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x128xf32>
    %126 = "ttir.div"(%122, %125) : (tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %127 = "ttir.typecast"(%126) <{conservative_folding = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xbf16>
    %128 = "ttir.reshape"(%78) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x128x64xbf16>) -> tensor<32x8x1x128x64xbf16>
    %129 = "ttir.broadcast"(%128) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x128x64xbf16>) -> tensor<32x8x4x128x64xbf16>
    %130 = "ttir.reshape"(%129) <{shape = [32 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x4x128x64xbf16>) -> tensor<32x32x128x64xbf16>
    %131 = "ttir.dot_general"(%127, %130) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<32x32x1x128xbf16>, tensor<32x32x128x64xbf16>) -> tensor<32x32x1x64xbf16>
    %132 = "ttir.reshape"(%131) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x32x1x64xbf16>) -> tensor<32x2048xbf16>
    %133 = "ttir.reshape"(%arg12) <{shape = [1 : i32, 2048 : i32, 2048 : i32]}> : (tensor<2048x2048xbf16>) -> tensor<1x2048x2048xbf16>
    %134 = "ttir.reshape"(%133) <{shape = [2048 : i32, 2048 : i32]}> : (tensor<1x2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %135 = "ttir.permute"(%134) <{permutation = array<i64: 1, 0>}> : (tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %136 = "ttir.dot_general"(%132, %135) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<32x2048xbf16>
    %137 = "ttir.reshape"(%136) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
    %138 = "ttir.add"(%26, %137) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %139 = "ttir.reshape"(%arg14) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %140 = "ttir.reshape"(%139) <{shape = [2048 : i32]}> : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %141 = "ttir.reshape"(%140) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %142 = "ttir.broadcast"(%141) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %143 = "ttir.typecast"(%138) <{conservative_folding = false}> : (tensor<32x1x2048xbf16>) -> tensor<32x1x2048xf32>
    %144 = "ttir.pow"(%143, %5) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
    %145 = "ttir.sum"(%144) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1xf32>
    %146 = "ttir.multiply"(%145, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %147 = "ttir.reshape"(%146) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %148 = "ttir.add"(%147, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %149 = "ttir.rsqrt"(%148) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %150 = "ttir.reshape"(%149) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
    %151 = "ttir.reshape"(%150) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %152 = "ttir.broadcast"(%151) <{broadcast_dimensions = array<i64: 1, 1, 2048>}> : (tensor<32x1x1xf32>) -> tensor<32x1x2048xf32>
    %153 = "ttir.multiply"(%143, %152) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
    %154 = "ttir.typecast"(%153) <{conservative_folding = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1x2048xbf16>
    %155 = "ttir.multiply"(%142, %154) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %156 = "ttir.reshape"(%155) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
    %157 = "ttir.reshape"(%arg15) <{shape = [1 : i32, 8192 : i32, 2048 : i32]}> : (tensor<8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %158 = "ttir.reshape"(%157) <{shape = [8192 : i32, 2048 : i32]}> : (tensor<1x8192x2048xbf16>) -> tensor<8192x2048xbf16>
    %159 = "ttir.permute"(%158) <{permutation = array<i64: 1, 0>}> : (tensor<8192x2048xbf16>) -> tensor<2048x8192xbf16>
    %160 = "ttir.dot_general"(%156, %159) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x8192xbf16>) -> tensor<32x8192xbf16>
    %161 = "ttir.reshape"(%160) <{shape = [32 : i32, 1 : i32, 8192 : i32]}> : (tensor<32x8192xbf16>) -> tensor<32x1x8192xbf16>
    %162 = "ttir.sigmoid"(%161) : (tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>
    %163 = "ttir.multiply"(%161, %162) : (tensor<32x1x8192xbf16>, tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>
    %164 = "ttir.reshape"(%arg11) <{shape = [1 : i32, 8192 : i32, 2048 : i32]}> : (tensor<8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %165 = "ttir.reshape"(%164) <{shape = [8192 : i32, 2048 : i32]}> : (tensor<1x8192x2048xbf16>) -> tensor<8192x2048xbf16>
    %166 = "ttir.permute"(%165) <{permutation = array<i64: 1, 0>}> : (tensor<8192x2048xbf16>) -> tensor<2048x8192xbf16>
    %167 = "ttir.dot_general"(%156, %166) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x8192xbf16>) -> tensor<32x8192xbf16>
    %168 = "ttir.reshape"(%167) <{shape = [32 : i32, 1 : i32, 8192 : i32]}> : (tensor<32x8192xbf16>) -> tensor<32x1x8192xbf16>
    %169 = "ttir.multiply"(%163, %168) : (tensor<32x1x8192xbf16>, tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>
    %170 = "ttir.reshape"(%169) <{shape = [32 : i32, 8192 : i32]}> : (tensor<32x1x8192xbf16>) -> tensor<32x8192xbf16>
    %171 = "ttir.reshape"(%arg10) <{shape = [1 : i32, 2048 : i32, 8192 : i32]}> : (tensor<2048x8192xbf16>) -> tensor<1x2048x8192xbf16>
    %172 = "ttir.reshape"(%171) <{shape = [2048 : i32, 8192 : i32]}> : (tensor<1x2048x8192xbf16>) -> tensor<2048x8192xbf16>
    %173 = "ttir.permute"(%172) <{permutation = array<i64: 1, 0>}> : (tensor<2048x8192xbf16>) -> tensor<8192x2048xbf16>
    %174 = "ttir.dot_general"(%170, %173) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x8192xbf16>, tensor<8192x2048xbf16>) -> tensor<32x2048xbf16>
    %175 = "ttir.reshape"(%174) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
    %176 = "ttir.add"(%138, %175) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %177 = "ttir.typecast"(%176) <{conservative_folding = false}> : (tensor<32x1x2048xbf16>) -> tensor<32x1x2048xf32>
    %178 = "ttir.pow"(%177, %5) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
    %179 = "ttir.sum"(%178) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1xf32>
    %180 = "ttir.multiply"(%179, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %181 = "ttir.reshape"(%180) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %182 = "ttir.add"(%181, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %183 = "ttir.rsqrt"(%182) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %184 = "ttir.reshape"(%183) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
    %185 = "ttir.reshape"(%184) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %186 = "ttir.broadcast"(%185) <{broadcast_dimensions = array<i64: 1, 1, 2048>}> : (tensor<32x1x1xf32>) -> tensor<32x1x2048xf32>
    %187 = "ttir.multiply"(%177, %186) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
    %188 = "ttir.typecast"(%187) <{conservative_folding = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1x2048xbf16>
    %189 = "ttir.multiply"(%82, %188) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %190 = "ttir.reshape"(%189) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
    %191 = "ttir.reshape"(%arg9) <{shape = [1 : i32, 128256 : i32, 2048 : i32]}> : (tensor<128256x2048xbf16>) -> tensor<1x128256x2048xbf16>
    %192 = "ttir.reshape"(%191) <{shape = [128256 : i32, 2048 : i32]}> : (tensor<1x128256x2048xbf16>) -> tensor<128256x2048xbf16>
    %193 = "ttir.permute"(%192) <{permutation = array<i64: 1, 0>}> : (tensor<128256x2048xbf16>) -> tensor<2048x128256xbf16>
    %194 = "ttir.dot_general"(%190, %193) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x128256xbf16>) -> tensor<32x128256xbf16>
    %195 = "ttir.reshape"(%194) <{shape = [32 : i32, 1 : i32, 128256 : i32]}> : (tensor<32x128256xbf16>) -> tensor<32x1x128256xbf16>
    return %71, %78, %195 : tensor<32x8x128x64xbf16>, tensor<32x8x128x64xbf16>, tensor<32x1x128256xbf16>
}
