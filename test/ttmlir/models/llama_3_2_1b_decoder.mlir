func.func @llama_3_2_1b_decoder(%arg0: tensor<1xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_1"} loc("p0.3"), %arg1: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_rotary_emb_inv_freq"} loc("p1.13"), %arg2: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_k_proj_weight"} loc("p2.31"), %arg3: tensor<32x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "embed_output"} loc("p3.39"), %arg4: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___input_layernorm_weight"} loc("p5.80"), %arg5: tensor<32x8x128x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_2"} loc("p6.120"), %arg6: tensor<512x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_v_proj_weight"} loc("p7.128"), %arg7: tensor<32x8x128x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_3"} loc("p8.148"), %arg8: tensor<2048x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___mlp_down_proj_weight"} loc("p10.165"), %arg9: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___mlp_up_proj_weight"} loc("p11.170"), %arg10: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_o_proj_weight"} loc("p12.179"), %arg11: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___self_attn_q_proj_weight"} loc("p13.229"), %arg12: tensor<2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___post_attention_layernorm_weight"} loc("p14.319"), %arg13: tensor<8192x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___model_layers__modules__0___mlp_gate_proj_weight"} loc("p15.328")) -> (tensor<32x8x128x64xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<32x8x128x64xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<32x1x2048xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
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
    %16 = "ttir.reshape"(%arg4) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %17 = "ttir.reshape"(%16) <{shape = [2048 : i32]}> : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %18 = "ttir.reshape"(%17) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %19 = "ttir.broadcast"(%18) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %20 = "ttir.reshape"(%arg3) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
    %21 = "ttir.typecast"(%20) <{conservative_folding = false}> : (tensor<32x1x2048xbf16>) -> tensor<32x1x2048xf32>
    %22 = "ttir.pow"(%21, %5) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
    %23 = "ttir.sum"(%22) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1xf32>
    %24 = "ttir.multiply"(%23, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %25 = "ttir.reshape"(%24) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %26 = "ttir.add"(%25, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %27 = "ttir.rsqrt"(%26) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %28 = "ttir.reshape"(%27) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
    %29 = "ttir.reshape"(%28) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %30 = "ttir.broadcast"(%29) <{broadcast_dimensions = array<i64: 1, 1, 2048>}> : (tensor<32x1x1xf32>) -> tensor<32x1x2048xf32>
    %31 = "ttir.multiply"(%21, %30) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
    %32 = "ttir.typecast"(%31) <{conservative_folding = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1x2048xbf16>
    %33 = "ttir.multiply"(%19, %32) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %34 = "ttir.reshape"(%33) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
    %35 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %36 = "ttir.reshape"(%35) <{shape = [512 : i32, 2048 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %37 = "ttir.permute"(%36) <{permutation = array<i64: 1, 0>}> : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %38 = "ttir.dot_general"(%34, %37) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x512xbf16>) -> tensor<32x512xbf16>
    %39 = "ttir.reshape"(%38) <{shape = [32 : i32, 8 : i32, 1 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<32x8x1x64xbf16>
    %40 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<32xbf16>) -> tensor<1x1x32xbf16>
    %41 = "ttir.reshape"(%40) <{shape = [1 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x32xbf16>) -> tensor<1x32x1xbf16>
    %42 = "ttir.typecast"(%41) <{conservative_folding = false}> : (tensor<1x32x1xbf16>) -> tensor<1x32x1xf32>
    %43 = "ttir.typecast"(%10) <{conservative_folding = false}> : (tensor<1x1x1xi64>) -> tensor<1x1x1xf32>
    %44 = "ttir.dot_general"(%42, %43) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<1x32x1xf32>, tensor<1x1x1xf32>) -> tensor<1x32x1xf32>
    %45 = "ttir.reshape"(%44) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x32x1xf32>) -> tensor<1x1x32xf32>
    %46 = "ttir.concat"(%45, %45) <{dim = 2 : si32}> : (tensor<1x1x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x64xf32>
    %47 = "ttir.cos"(%46) : (tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
    %48 = "ttir.typecast"(%47) <{conservative_folding = false}> : (tensor<1x1x64xf32>) -> tensor<1x1x64xbf16>
    %49 = "ttir.reshape"(%48) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x1x64xbf16>) -> tensor<1x64xbf16>
    %50 = "ttir.reshape"(%49) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x1x64xbf16>
    %51 = "ttir.broadcast"(%50) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %52 = "ttir.multiply"(%39, %51) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %53 = "ttir.slice_static"(%39) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %54 = "ttir.neg"(%53) : (tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %55 = "ttir.slice_static"(%39) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %56 = "ttir.concat"(%54, %55) <{dim = 3 : si32}> : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x64xbf16>
    %57 = "ttir.sin"(%46) : (tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
    %58 = "ttir.typecast"(%57) <{conservative_folding = false}> : (tensor<1x1x64xf32>) -> tensor<1x1x64xbf16>
    %59 = "ttir.reshape"(%58) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x1x64xbf16>) -> tensor<1x64xbf16>
    %60 = "ttir.reshape"(%59) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x1x64xbf16>
    %61 = "ttir.broadcast"(%60) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %62 = "ttir.multiply"(%56, %61) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %63 = "ttir.add"(%52, %62) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %64 = "ttir.permute"(%63) <{permutation = array<i64: 2, 1, 0, 3>}> : (tensor<32x8x1x64xbf16>) -> tensor<1x8x32x64xbf16>
    %65 = "ttir.update_cache"(%arg5, %64, %arg0) <{batch_offset = 0 : i32}> : (tensor<32x8x128x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1xi64>) -> tensor<32x8x128x64xbf16>
    %66 = "ttir.reshape"(%arg6) <{shape = [1 : i32, 512 : i32, 2048 : i32]}> : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %67 = "ttir.reshape"(%66) <{shape = [512 : i32, 2048 : i32]}> : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %68 = "ttir.permute"(%67) <{permutation = array<i64: 1, 0>}> : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %69 = "ttir.dot_general"(%34, %68) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x512xbf16>) -> tensor<32x512xbf16>
    %70 = "ttir.reshape"(%69) <{shape = [32 : i32, 8 : i32, 1 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<32x8x1x64xbf16>
    %71 = "ttir.permute"(%70) <{permutation = array<i64: 2, 1, 0, 3>}> : (tensor<32x8x1x64xbf16>) -> tensor<1x8x32x64xbf16>
    %72 = "ttir.update_cache"(%arg7, %71, %arg0) <{batch_offset = 0 : i32}> : (tensor<32x8x128x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1xi64>) -> tensor<32x8x128x64xbf16>
    %73 = "ttir.reshape"(%arg11) <{shape = [1 : i32, 2048 : i32, 2048 : i32]}> : (tensor<2048x2048xbf16>) -> tensor<1x2048x2048xbf16>
    %74 = "ttir.reshape"(%73) <{shape = [2048 : i32, 2048 : i32]}> : (tensor<1x2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %75 = "ttir.permute"(%74) <{permutation = array<i64: 1, 0>}> : (tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %76 = "ttir.dot_general"(%34, %75) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<32x2048xbf16>
    %77 = "ttir.reshape"(%76) <{shape = [32 : i32, 32 : i32, 1 : i32, 64 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x32x1x64xbf16>
    %78 = "ttir.reshape"(%49) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x1x64xbf16>
    %79 = "ttir.broadcast"(%78) <{broadcast_dimensions = array<i64: 32, 32, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x32x1x64xbf16>
    %80 = "ttir.multiply"(%77, %79) : (tensor<32x32x1x64xbf16>, tensor<32x32x1x64xbf16>) -> tensor<32x32x1x64xbf16>
    %81 = "ttir.slice_static"(%77) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [32 : i32, 32 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1x64xbf16>) -> tensor<32x32x1x32xbf16>
    %82 = "ttir.neg"(%81) : (tensor<32x32x1x32xbf16>) -> tensor<32x32x1x32xbf16>
    %83 = "ttir.slice_static"(%77) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 32 : i32, 1 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1x64xbf16>) -> tensor<32x32x1x32xbf16>
    %84 = "ttir.concat"(%82, %83) <{dim = 3 : si32}> : (tensor<32x32x1x32xbf16>, tensor<32x32x1x32xbf16>) -> tensor<32x32x1x64xbf16>
    %85 = "ttir.reshape"(%59) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64xbf16>) -> tensor<1x1x1x64xbf16>
    %86 = "ttir.broadcast"(%85) <{broadcast_dimensions = array<i64: 32, 32, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x32x1x64xbf16>
    %87 = "ttir.multiply"(%84, %86) : (tensor<32x32x1x64xbf16>, tensor<32x32x1x64xbf16>) -> tensor<32x32x1x64xbf16>
    %88 = "ttir.add"(%80, %87) : (tensor<32x32x1x64xbf16>, tensor<32x32x1x64xbf16>) -> tensor<32x32x1x64xbf16>
    %89 = "ttir.reshape"(%65) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x128x64xbf16>) -> tensor<32x8x1x128x64xbf16>
    %90 = "ttir.broadcast"(%89) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x128x64xbf16>) -> tensor<32x8x4x128x64xbf16>
    %91 = "ttir.reshape"(%90) <{shape = [32 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x4x128x64xbf16>) -> tensor<32x32x128x64xbf16>
    %92 = "ttir.permute"(%91) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<32x32x128x64xbf16>) -> tensor<32x32x64x128xbf16>
    %93 = "ttir.dot_general"(%88, %92) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<32x32x1x64xbf16>, tensor<32x32x64x128xbf16>) -> tensor<32x32x1x128xbf16>
    %94 = "ttir.multiply"(%93, %2) : (tensor<32x32x1x128xbf16>, tensor<32x32x1x128xbf16>) -> tensor<32x32x1x128xbf16>
    %95 = "ttir.reshape"(%11) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi64>) -> tensor<1x1xi64>
    %96 = "ttir.broadcast"(%95) <{broadcast_dimensions = array<i64: 1, 128>}> : (tensor<1x1xi64>) -> tensor<1x128xi64>
    %97 = "ttir.gt"(%0, %96) : (tensor<1x128xi64>, tensor<1x128xi64>) -> tensor<1x128xi1>
    %98 = "ttir.typecast"(%97) <{conservative_folding = false}> : (tensor<1x128xi1>) -> tensor<1x128xbf16>
    %99 = "ttir.multiply"(%98, %1) : (tensor<1x128xbf16>, tensor<1x128xbf16>) -> tensor<1x128xbf16>
    %100 = "ttir.reshape"(%99) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xbf16>) -> tensor<1x1x128xbf16>
    %101 = "ttir.reshape"(%100) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    %102 = "ttir.broadcast"(%101) <{broadcast_dimensions = array<i64: 32, 1, 1, 1>}> : (tensor<1x1x1x128xbf16>) -> tensor<32x1x1x128xbf16>
    %103 = "ttir.reshape"(%102) <{shape = [32 : i32, 1 : i32, 128 : i32]}> : (tensor<32x1x1x128xbf16>) -> tensor<32x1x128xbf16>
    %104 = "ttir.reshape"(%103) <{shape = [32 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<32x1x128xbf16>) -> tensor<32x1x1x128xbf16>
    %105 = "ttir.broadcast"(%104) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<32x1x1x128xbf16>) -> tensor<32x32x1x128xbf16>
    %106 = "ttir.add"(%94, %105) : (tensor<32x32x1x128xbf16>, tensor<32x32x1x128xbf16>) -> tensor<32x32x1x128xbf16>
    %107 = "ttir.typecast"(%106) <{conservative_folding = false}> : (tensor<32x32x1x128xbf16>) -> tensor<32x32x1x128xf32>
    %108 = "ttir.max"(%107) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1xf32>
    %109 = "ttir.reshape"(%108) <{shape = [32 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1xf32>) -> tensor<32x32x1x1xf32>
    %110 = "ttir.broadcast"(%109) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x128xf32>
    %111 = "ttir.subtract"(%107, %110) : (tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %112 = "ttir.exp"(%111) : (tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %113 = "ttir.sum"(%112) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1xf32>
    %114 = "ttir.reshape"(%113) <{shape = [32 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1xf32>) -> tensor<32x32x1x1xf32>
    %115 = "ttir.broadcast"(%114) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x128xf32>
    %116 = "ttir.div"(%112, %115) : (tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %117 = "ttir.typecast"(%116) <{conservative_folding = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xbf16>
    %118 = "ttir.reshape"(%72) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x128x64xbf16>) -> tensor<32x8x1x128x64xbf16>
    %119 = "ttir.broadcast"(%118) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x128x64xbf16>) -> tensor<32x8x4x128x64xbf16>
    %120 = "ttir.reshape"(%119) <{shape = [32 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x4x128x64xbf16>) -> tensor<32x32x128x64xbf16>
    %121 = "ttir.dot_general"(%117, %120) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<32x32x1x128xbf16>, tensor<32x32x128x64xbf16>) -> tensor<32x32x1x64xbf16>
    %122 = "ttir.reshape"(%121) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x32x1x64xbf16>) -> tensor<32x2048xbf16>
    %123 = "ttir.reshape"(%arg10) <{shape = [1 : i32, 2048 : i32, 2048 : i32]}> : (tensor<2048x2048xbf16>) -> tensor<1x2048x2048xbf16>
    %124 = "ttir.reshape"(%123) <{shape = [2048 : i32, 2048 : i32]}> : (tensor<1x2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %125 = "ttir.permute"(%124) <{permutation = array<i64: 1, 0>}> : (tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %126 = "ttir.dot_general"(%122, %125) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<32x2048xbf16>
    %127 = "ttir.reshape"(%126) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
    %128 = "ttir.add"(%20, %127) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %129 = "ttir.reshape"(%arg12) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %130 = "ttir.reshape"(%129) <{shape = [2048 : i32]}> : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %131 = "ttir.reshape"(%130) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %132 = "ttir.broadcast"(%131) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %133 = "ttir.typecast"(%128) <{conservative_folding = false}> : (tensor<32x1x2048xbf16>) -> tensor<32x1x2048xf32>
    %134 = "ttir.pow"(%133, %5) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
    %135 = "ttir.sum"(%134) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1xf32>
    %136 = "ttir.multiply"(%135, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %137 = "ttir.reshape"(%136) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %138 = "ttir.add"(%137, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %139 = "ttir.rsqrt"(%138) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %140 = "ttir.reshape"(%139) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
    %141 = "ttir.reshape"(%140) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %142 = "ttir.broadcast"(%141) <{broadcast_dimensions = array<i64: 1, 1, 2048>}> : (tensor<32x1x1xf32>) -> tensor<32x1x2048xf32>
    %143 = "ttir.multiply"(%133, %142) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
    %144 = "ttir.typecast"(%143) <{conservative_folding = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1x2048xbf16>
    %145 = "ttir.multiply"(%132, %144) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    %146 = "ttir.reshape"(%145) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
    %147 = "ttir.reshape"(%arg13) <{shape = [1 : i32, 8192 : i32, 2048 : i32]}> : (tensor<8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %148 = "ttir.reshape"(%147) <{shape = [8192 : i32, 2048 : i32]}> : (tensor<1x8192x2048xbf16>) -> tensor<8192x2048xbf16>
    %149 = "ttir.permute"(%148) <{permutation = array<i64: 1, 0>}> : (tensor<8192x2048xbf16>) -> tensor<2048x8192xbf16>
    %150 = "ttir.dot_general"(%146, %149) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x8192xbf16>) -> tensor<32x8192xbf16>
    %151 = "ttir.reshape"(%150) <{shape = [32 : i32, 1 : i32, 8192 : i32]}> : (tensor<32x8192xbf16>) -> tensor<32x1x8192xbf16>
    %152 = "ttir.sigmoid"(%151) : (tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>
    %153 = "ttir.multiply"(%151, %152) : (tensor<32x1x8192xbf16>, tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>
    %154 = "ttir.reshape"(%arg9) <{shape = [1 : i32, 8192 : i32, 2048 : i32]}> : (tensor<8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %155 = "ttir.reshape"(%154) <{shape = [8192 : i32, 2048 : i32]}> : (tensor<1x8192x2048xbf16>) -> tensor<8192x2048xbf16>
    %156 = "ttir.permute"(%155) <{permutation = array<i64: 1, 0>}> : (tensor<8192x2048xbf16>) -> tensor<2048x8192xbf16>
    %157 = "ttir.dot_general"(%146, %156) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x2048xbf16>, tensor<2048x8192xbf16>) -> tensor<32x8192xbf16>
    %158 = "ttir.reshape"(%157) <{shape = [32 : i32, 1 : i32, 8192 : i32]}> : (tensor<32x8192xbf16>) -> tensor<32x1x8192xbf16>
    %159 = "ttir.multiply"(%153, %158) : (tensor<32x1x8192xbf16>, tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>
    %160 = "ttir.reshape"(%159) <{shape = [32 : i32, 8192 : i32]}> : (tensor<32x1x8192xbf16>) -> tensor<32x8192xbf16>
    %161 = "ttir.reshape"(%arg8) <{shape = [1 : i32, 2048 : i32, 8192 : i32]}> : (tensor<2048x8192xbf16>) -> tensor<1x2048x8192xbf16>
    %162 = "ttir.reshape"(%161) <{shape = [2048 : i32, 8192 : i32]}> : (tensor<1x2048x8192xbf16>) -> tensor<2048x8192xbf16>
    %163 = "ttir.permute"(%162) <{permutation = array<i64: 1, 0>}> : (tensor<2048x8192xbf16>) -> tensor<8192x2048xbf16>
    %164 = "ttir.dot_general"(%160, %163) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x8192xbf16>, tensor<8192x2048xbf16>) -> tensor<32x2048xbf16>
    %165 = "ttir.reshape"(%164) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
    %166 = "ttir.add"(%128, %165) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
    return %65, %72, %166 : tensor<32x8x128x64xbf16>, tensor<32x8x128x64xbf16>, tensor<32x1x2048xbf16>
}
