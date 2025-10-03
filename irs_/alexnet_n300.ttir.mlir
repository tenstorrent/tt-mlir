module @jit__lambda attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func public @main(%arg0: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<11x11x3x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<5x5x64x192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg4: tensor<384xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg5: tensor<3x3x192x384xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg6: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg7: tensor<3x3x384x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg8: tensor<256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg9: tensor<3x3x256x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg10: tensor<4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg11: tensor<6400x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg12: tensor<4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg13: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg14: tensor<1000xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg15: tensor<4096x1000xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg16: tensor<8x224x224x3xi32> {ttcore.argument_type = #ttcore.argument_type<input>}) -> (tensor<8x1000xbf16> {jax.result_info = "result"}) {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<64xbf16>) -> tensor<64xbf16>
    %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<11x11x3x64xbf16>) -> tensor<11x11x3x64xbf16>
    %2 = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<192xbf16>) -> tensor<192xbf16>
    %3 = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<5x5x64x192xbf16>) -> tensor<5x5x64x192xbf16>
    %4 = "ttir.mesh_shard"(%arg4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<384xbf16>) -> tensor<384xbf16>
    %5 = "ttir.mesh_shard"(%arg5) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<3x3x192x384xbf16>) -> tensor<3x3x192x384xbf16>
    %6 = "ttir.mesh_shard"(%arg6) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<256xbf16>) -> tensor<256xbf16>
    %7 = "ttir.mesh_shard"(%arg7) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<3x3x384x256xbf16>) -> tensor<3x3x384x256xbf16>
    %8 = "ttir.mesh_shard"(%arg8) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<256xbf16>) -> tensor<256xbf16>
    %9 = "ttir.mesh_shard"(%arg9) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xbf16>
    %10 = "ttir.mesh_shard"(%arg10) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<4096xbf16>) -> tensor<2048xbf16>
    %11 = "ttir.mesh_shard"(%arg11) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<6400x4096xbf16>) -> tensor<6400x2048xbf16>
    %12 = "ttir.mesh_shard"(%arg12) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<4096xbf16>) -> tensor<2048xbf16>
    %13 = "ttir.mesh_shard"(%arg13) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<4096x4096xbf16>) -> tensor<4096x2048xbf16>
    %14 = "ttir.mesh_shard"(%arg14) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1000xbf16>) -> tensor<500xbf16>
    %15 = "ttir.mesh_shard"(%arg15) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<4096x1000xbf16>) -> tensor<4096x500xbf16>
    %16 = "ttir.mesh_shard"(%arg16) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<8x224x224x3xi32>) -> tensor<4x224x224x3xi32>
    %17 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %18 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %19 = "ttir.constant"() <{value = dense<0xFF80> : tensor<bf16>}> : () -> tensor<bf16>
    %20 = ttir.empty() : tensor<4x224x224x3xbf16>
    %21 = "ttir.typecast"(%16, %20) <{conservative_folding = false}> : (tensor<4x224x224x3xi32>, tensor<4x224x224x3xbf16>) -> tensor<4x224x224x3xbf16>
    %22 = ttir.empty() : tensor<4x54x54x64xbf16>
    %23 = "ttir.convolution"(%21, %1, %22) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 3, input_spatial_dimensions = 1x2, kernel_output_feature = 3, kernel_input_feature = 2, kernel_spatial_dimensions = 0x1, output_batch = 0, output_feature = 3, output_spatial_dimensions = 1x2>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 4, 4>}> : (tensor<4x224x224x3xbf16>, tensor<11x11x3x64xbf16>, tensor<4x54x54x64xbf16>) -> tensor<4x54x54x64xbf16>
    %24 = ttir.empty() : tensor<1x1x1x64xbf16>
    %25 = "ttir.reshape"(%0, %24) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xbf16>
    %26 = ttir.empty() : tensor<4x54x54x64xbf16>
    %27 = "ttir.broadcast"(%25, %26) <{broadcast_dimensions = array<i64: 4, 54, 54, 1>}> : (tensor<1x1x1x64xbf16>, tensor<4x54x54x64xbf16>) -> tensor<4x54x54x64xbf16>
    %28 = ttir.empty() : tensor<4x54x54x64xbf16>
    %29 = "ttir.add"(%23, %27, %28) : (tensor<4x54x54x64xbf16>, tensor<4x54x54x64xbf16>, tensor<4x54x54x64xbf16>) -> tensor<4x54x54x64xbf16>
    %30 = ttir.empty() : tensor<1x1x1x1xbf16>
    %31 = "ttir.reshape"(%17, %30) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1x1x1xbf16>) -> tensor<1x1x1x1xbf16>
    %32 = ttir.empty() : tensor<4x54x54x64xbf16>
    %33 = "ttir.broadcast"(%31, %32) <{broadcast_dimensions = array<i64: 4, 54, 54, 64>}> : (tensor<1x1x1x1xbf16>, tensor<4x54x54x64xbf16>) -> tensor<4x54x54x64xbf16>
    %34 = ttir.empty() : tensor<4x54x54x64xbf16>
    %35 = "ttir.maximum"(%29, %33, %34) : (tensor<4x54x54x64xbf16>, tensor<4x54x54x64xbf16>, tensor<4x54x54x64xbf16>) -> tensor<4x54x54x64xbf16>
    %36 = ttir.empty() : tensor<bf16>
    %37 = "ttir.broadcast"(%19, %36) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
    %38 = ttir.empty() : tensor<4x26x26x64xbf16>
    %39 = "ttir.pooling"(%35, %38) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> : (tensor<4x54x54x64xbf16>, tensor<4x26x26x64xbf16>) -> tensor<4x26x26x64xbf16>
    %40 = ttir.empty() : tensor<4x26x26x192xbf16>
    %41 = "ttir.convolution"(%39, %3, %40) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 3, input_spatial_dimensions = 1x2, kernel_output_feature = 3, kernel_input_feature = 2, kernel_spatial_dimensions = 0x1, output_batch = 0, output_feature = 3, output_spatial_dimensions = 1x2>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 2, 2, 2, 2>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<4x26x26x64xbf16>, tensor<5x5x64x192xbf16>, tensor<4x26x26x192xbf16>) -> tensor<4x26x26x192xbf16>
    %42 = ttir.empty() : tensor<1x1x1x192xbf16>
    %43 = "ttir.reshape"(%2, %42) <{shape = [1 : i32, 1 : i32, 1 : i32, 192 : i32]}> : (tensor<192xbf16>, tensor<1x1x1x192xbf16>) -> tensor<1x1x1x192xbf16>
    %44 = ttir.empty() : tensor<4x26x26x192xbf16>
    %45 = "ttir.broadcast"(%43, %44) <{broadcast_dimensions = array<i64: 4, 26, 26, 1>}> : (tensor<1x1x1x192xbf16>, tensor<4x26x26x192xbf16>) -> tensor<4x26x26x192xbf16>
    %46 = ttir.empty() : tensor<4x26x26x192xbf16>
    %47 = "ttir.add"(%41, %45, %46) : (tensor<4x26x26x192xbf16>, tensor<4x26x26x192xbf16>, tensor<4x26x26x192xbf16>) -> tensor<4x26x26x192xbf16>
    %48 = ttir.empty() : tensor<1x1x1x1xbf16>
    %49 = "ttir.reshape"(%17, %48) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1x1x1xbf16>) -> tensor<1x1x1x1xbf16>
    %50 = ttir.empty() : tensor<4x26x26x192xbf16>
    %51 = "ttir.broadcast"(%49, %50) <{broadcast_dimensions = array<i64: 4, 26, 26, 192>}> : (tensor<1x1x1x1xbf16>, tensor<4x26x26x192xbf16>) -> tensor<4x26x26x192xbf16>
    %52 = ttir.empty() : tensor<4x26x26x192xbf16>
    %53 = "ttir.maximum"(%47, %51, %52) : (tensor<4x26x26x192xbf16>, tensor<4x26x26x192xbf16>, tensor<4x26x26x192xbf16>) -> tensor<4x26x26x192xbf16>
    %54 = ttir.empty() : tensor<bf16>
    %55 = "ttir.broadcast"(%19, %54) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
    %56 = ttir.empty() : tensor<4x12x12x192xbf16>
    %57 = "ttir.pooling"(%53, %56) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> : (tensor<4x26x26x192xbf16>, tensor<4x12x12x192xbf16>) -> tensor<4x12x12x192xbf16>
    %58 = ttir.empty() : tensor<4x12x12x384xbf16>
    %59 = "ttir.convolution"(%57, %5, %58) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 3, input_spatial_dimensions = 1x2, kernel_output_feature = 3, kernel_input_feature = 2, kernel_spatial_dimensions = 0x1, output_batch = 0, output_feature = 3, output_spatial_dimensions = 1x2>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<4x12x12x192xbf16>, tensor<3x3x192x384xbf16>, tensor<4x12x12x384xbf16>) -> tensor<4x12x12x384xbf16>
    %60 = ttir.empty() : tensor<1x1x1x384xbf16>
    %61 = "ttir.reshape"(%4, %60) <{shape = [1 : i32, 1 : i32, 1 : i32, 384 : i32]}> : (tensor<384xbf16>, tensor<1x1x1x384xbf16>) -> tensor<1x1x1x384xbf16>
    %62 = ttir.empty() : tensor<4x12x12x384xbf16>
    %63 = "ttir.broadcast"(%61, %62) <{broadcast_dimensions = array<i64: 4, 12, 12, 1>}> : (tensor<1x1x1x384xbf16>, tensor<4x12x12x384xbf16>) -> tensor<4x12x12x384xbf16>
    %64 = ttir.empty() : tensor<4x12x12x384xbf16>
    %65 = "ttir.add"(%59, %63, %64) : (tensor<4x12x12x384xbf16>, tensor<4x12x12x384xbf16>, tensor<4x12x12x384xbf16>) -> tensor<4x12x12x384xbf16>
    %66 = ttir.empty() : tensor<1x1x1x1xbf16>
    %67 = "ttir.reshape"(%17, %66) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1x1x1xbf16>) -> tensor<1x1x1x1xbf16>
    %68 = ttir.empty() : tensor<4x12x12x384xbf16>
    %69 = "ttir.broadcast"(%67, %68) <{broadcast_dimensions = array<i64: 4, 12, 12, 384>}> : (tensor<1x1x1x1xbf16>, tensor<4x12x12x384xbf16>) -> tensor<4x12x12x384xbf16>
    %70 = ttir.empty() : tensor<4x12x12x384xbf16>
    %71 = "ttir.maximum"(%65, %69, %70) : (tensor<4x12x12x384xbf16>, tensor<4x12x12x384xbf16>, tensor<4x12x12x384xbf16>) -> tensor<4x12x12x384xbf16>
    %72 = ttir.empty() : tensor<4x12x12x256xbf16>
    %73 = "ttir.convolution"(%71, %7, %72) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 3, input_spatial_dimensions = 1x2, kernel_output_feature = 3, kernel_input_feature = 2, kernel_spatial_dimensions = 0x1, output_batch = 0, output_feature = 3, output_spatial_dimensions = 1x2>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<4x12x12x384xbf16>, tensor<3x3x384x256xbf16>, tensor<4x12x12x256xbf16>) -> tensor<4x12x12x256xbf16>
    %74 = ttir.empty() : tensor<1x1x1x256xbf16>
    %75 = "ttir.reshape"(%6, %74) <{shape = [1 : i32, 1 : i32, 1 : i32, 256 : i32]}> : (tensor<256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x1x1x256xbf16>
    %76 = ttir.empty() : tensor<4x12x12x256xbf16>
    %77 = "ttir.broadcast"(%75, %76) <{broadcast_dimensions = array<i64: 4, 12, 12, 1>}> : (tensor<1x1x1x256xbf16>, tensor<4x12x12x256xbf16>) -> tensor<4x12x12x256xbf16>
    %78 = ttir.empty() : tensor<4x12x12x256xbf16>
    %79 = "ttir.add"(%73, %77, %78) : (tensor<4x12x12x256xbf16>, tensor<4x12x12x256xbf16>, tensor<4x12x12x256xbf16>) -> tensor<4x12x12x256xbf16>
    %80 = ttir.empty() : tensor<1x1x1x1xbf16>
    %81 = "ttir.reshape"(%17, %80) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1x1x1xbf16>) -> tensor<1x1x1x1xbf16>
    %82 = ttir.empty() : tensor<4x12x12x256xbf16>
    %83 = "ttir.broadcast"(%81, %82) <{broadcast_dimensions = array<i64: 4, 12, 12, 256>}> : (tensor<1x1x1x1xbf16>, tensor<4x12x12x256xbf16>) -> tensor<4x12x12x256xbf16>
    %84 = ttir.empty() : tensor<4x12x12x256xbf16>
    %85 = "ttir.maximum"(%79, %83, %84) : (tensor<4x12x12x256xbf16>, tensor<4x12x12x256xbf16>, tensor<4x12x12x256xbf16>) -> tensor<4x12x12x256xbf16>
    %86 = ttir.empty() : tensor<4x12x12x256xbf16>
    %87 = "ttir.convolution"(%85, %9, %86) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 3, input_spatial_dimensions = 1x2, kernel_output_feature = 3, kernel_input_feature = 2, kernel_spatial_dimensions = 0x1, output_batch = 0, output_feature = 3, output_spatial_dimensions = 1x2>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<4x12x12x256xbf16>, tensor<3x3x256x256xbf16>, tensor<4x12x12x256xbf16>) -> tensor<4x12x12x256xbf16>
    %88 = ttir.empty() : tensor<1x1x1x256xbf16>
    %89 = "ttir.reshape"(%8, %88) <{shape = [1 : i32, 1 : i32, 1 : i32, 256 : i32]}> : (tensor<256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x1x1x256xbf16>
    %90 = ttir.empty() : tensor<4x12x12x256xbf16>
    %91 = "ttir.broadcast"(%89, %90) <{broadcast_dimensions = array<i64: 4, 12, 12, 1>}> : (tensor<1x1x1x256xbf16>, tensor<4x12x12x256xbf16>) -> tensor<4x12x12x256xbf16>
    %92 = ttir.empty() : tensor<4x12x12x256xbf16>
    %93 = "ttir.add"(%87, %91, %92) : (tensor<4x12x12x256xbf16>, tensor<4x12x12x256xbf16>, tensor<4x12x12x256xbf16>) -> tensor<4x12x12x256xbf16>
    %94 = ttir.empty() : tensor<1x1x1x1xbf16>
    %95 = "ttir.reshape"(%17, %94) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1x1x1xbf16>) -> tensor<1x1x1x1xbf16>
    %96 = ttir.empty() : tensor<4x12x12x256xbf16>
    %97 = "ttir.broadcast"(%95, %96) <{broadcast_dimensions = array<i64: 4, 12, 12, 256>}> : (tensor<1x1x1x1xbf16>, tensor<4x12x12x256xbf16>) -> tensor<4x12x12x256xbf16>
    %98 = ttir.empty() : tensor<4x12x12x256xbf16>
    %99 = "ttir.maximum"(%93, %97, %98) : (tensor<4x12x12x256xbf16>, tensor<4x12x12x256xbf16>, tensor<4x12x12x256xbf16>) -> tensor<4x12x12x256xbf16>
    %100 = ttir.empty() : tensor<bf16>
    %101 = "ttir.broadcast"(%19, %100) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
    %102 = ttir.empty() : tensor<4x5x5x256xbf16>
    %103 = "ttir.pooling"(%99, %102) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> : (tensor<4x12x12x256xbf16>, tensor<4x5x5x256xbf16>) -> tensor<4x5x5x256xbf16>
    %104 = ttir.empty() : tensor<4x6400xbf16>
    %105 = "ttir.reshape"(%103, %104) <{shape = [4 : i32, 6400 : i32]}> : (tensor<4x5x5x256xbf16>, tensor<4x6400xbf16>) -> tensor<4x6400xbf16>
    %106 = ttir.empty() : tensor<8x6400xbf16>
    %107 = "ttir.all_gather"(%105, %106) <{all_gather_dim = 0 : si32, cluster_axis = 1 : ui32}> : (tensor<4x6400xbf16>, tensor<8x6400xbf16>) -> tensor<8x6400xbf16>
    %108 = "ttir.dot_general"(%107, %11) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x6400xbf16>, tensor<6400x2048xbf16>) -> tensor<8x2048xbf16>
    %109 = ttir.empty() : tensor<1x2048xbf16>
    %110 = "ttir.reshape"(%10, %109) <{shape = [1 : i32, 2048 : i32]}> : (tensor<2048xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
    %111 = ttir.empty() : tensor<8x2048xbf16>
    %112 = "ttir.broadcast"(%110, %111) <{broadcast_dimensions = array<i64: 8, 1>}> : (tensor<1x2048xbf16>, tensor<8x2048xbf16>) -> tensor<8x2048xbf16>
    %113 = ttir.empty() : tensor<8x2048xbf16>
    %114 = "ttir.add"(%108, %112, %113) : (tensor<8x2048xbf16>, tensor<8x2048xbf16>, tensor<8x2048xbf16>) -> tensor<8x2048xbf16>
    %115 = ttir.empty() : tensor<1x1xbf16>
    %116 = "ttir.reshape"(%17, %115) <{shape = [1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %117 = ttir.empty() : tensor<8x2048xbf16>
    %118 = "ttir.broadcast"(%116, %117) <{broadcast_dimensions = array<i64: 8, 2048>}> : (tensor<1x1xbf16>, tensor<8x2048xbf16>) -> tensor<8x2048xbf16>
    %119 = ttir.empty() : tensor<8x2048xbf16>
    %120 = "ttir.maximum"(%114, %118, %119) : (tensor<8x2048xbf16>, tensor<8x2048xbf16>, tensor<8x2048xbf16>) -> tensor<8x2048xbf16>
    %121 = ttir.empty() : tensor<8x4096xbf16>
    %122 = "ttir.all_gather"(%120, %121) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<8x2048xbf16>, tensor<8x4096xbf16>) -> tensor<8x4096xbf16>
    %123 = "ttir.dot_general"(%122, %13) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x4096xbf16>, tensor<4096x2048xbf16>) -> tensor<8x2048xbf16>
    %124 = ttir.empty() : tensor<1x2048xbf16>
    %125 = "ttir.reshape"(%12, %124) <{shape = [1 : i32, 2048 : i32]}> : (tensor<2048xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
    %126 = ttir.empty() : tensor<8x2048xbf16>
    %127 = "ttir.broadcast"(%125, %126) <{broadcast_dimensions = array<i64: 8, 1>}> : (tensor<1x2048xbf16>, tensor<8x2048xbf16>) -> tensor<8x2048xbf16>
    %128 = ttir.empty() : tensor<8x2048xbf16>
    %129 = "ttir.add"(%123, %127, %128) : (tensor<8x2048xbf16>, tensor<8x2048xbf16>, tensor<8x2048xbf16>) -> tensor<8x2048xbf16>
    %130 = ttir.empty() : tensor<1x1xbf16>
    %131 = "ttir.reshape"(%17, %130) <{shape = [1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %132 = ttir.empty() : tensor<8x2048xbf16>
    %133 = "ttir.broadcast"(%131, %132) <{broadcast_dimensions = array<i64: 8, 2048>}> : (tensor<1x1xbf16>, tensor<8x2048xbf16>) -> tensor<8x2048xbf16>
    %134 = ttir.empty() : tensor<8x2048xbf16>
    %135 = "ttir.maximum"(%129, %133, %134) : (tensor<8x2048xbf16>, tensor<8x2048xbf16>, tensor<8x2048xbf16>) -> tensor<8x2048xbf16>
    %136 = ttir.empty() : tensor<8x4096xbf16>
    %137 = "ttir.all_gather"(%135, %136) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<8x2048xbf16>, tensor<8x4096xbf16>) -> tensor<8x4096xbf16>
    %138 = "ttir.dot_general"(%137, %15) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<8x4096xbf16>, tensor<4096x500xbf16>) -> tensor<8x500xbf16>
    %139 = ttir.empty() : tensor<1x500xbf16>
    %140 = "ttir.reshape"(%14, %139) <{shape = [1 : i32, 500 : i32]}> : (tensor<500xbf16>, tensor<1x500xbf16>) -> tensor<1x500xbf16>
    %141 = ttir.empty() : tensor<8x500xbf16>
    %142 = "ttir.broadcast"(%140, %141) <{broadcast_dimensions = array<i64: 8, 1>}> : (tensor<1x500xbf16>, tensor<8x500xbf16>) -> tensor<8x500xbf16>
    %143 = ttir.empty() : tensor<8x500xbf16>
    %144 = "ttir.add"(%138, %142, %143) : (tensor<8x500xbf16>, tensor<8x500xbf16>, tensor<8x500xbf16>) -> tensor<8x500xbf16>
    %145 = ttir.empty() : tensor<8x1000xbf16>
    %146 = "ttir.all_gather"(%144, %145) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<8x500xbf16>, tensor<8x1000xbf16>) -> tensor<8x1000xbf16>
    %147 = ttir.empty() : tensor<8xbf16>
    %148 = "ttir.max"(%146, %147) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<8x1000xbf16>, tensor<8xbf16>) -> tensor<8xbf16>
    %149 = ttir.empty() : tensor<1xbf16>
    %150 = "ttir.reshape"(%19, %149) <{shape = [1 : i32]}> : (tensor<bf16>, tensor<1xbf16>) -> tensor<1xbf16>
    %151 = ttir.empty() : tensor<8xbf16>
    %152 = "ttir.broadcast"(%150, %151) <{broadcast_dimensions = array<i64: 8>}> : (tensor<1xbf16>, tensor<8xbf16>) -> tensor<8xbf16>
    %153 = ttir.empty() : tensor<8xbf16>
    %154 = "ttir.maximum"(%152, %148, %153) : (tensor<8xbf16>, tensor<8xbf16>, tensor<8xbf16>) -> tensor<8xbf16>
    %155 = ttir.empty() : tensor<8x1xbf16>
    %156 = "ttir.reshape"(%154, %155) <{shape = [8 : i32, 1 : i32]}> : (tensor<8xbf16>, tensor<8x1xbf16>) -> tensor<8x1xbf16>
    %157 = ttir.empty() : tensor<8x1xbf16>
    %158 = "ttir.broadcast"(%156, %157) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<8x1xbf16>, tensor<8x1xbf16>) -> tensor<8x1xbf16>
    %159 = ttir.empty() : tensor<8x1000xbf16>
    %160 = "ttir.broadcast"(%158, %159) <{broadcast_dimensions = array<i64: 1, 1000>}> : (tensor<8x1xbf16>, tensor<8x1000xbf16>) -> tensor<8x1000xbf16>
    %161 = ttir.empty() : tensor<8x1000xbf16>
    %162 = "ttir.subtract"(%146, %160, %161) : (tensor<8x1000xbf16>, tensor<8x1000xbf16>, tensor<8x1000xbf16>) -> tensor<8x1000xbf16>
    %163 = ttir.empty() : tensor<8x1000xbf16>
    %164 = "ttir.exp"(%162, %163) : (tensor<8x1000xbf16>, tensor<8x1000xbf16>) -> tensor<8x1000xbf16>
    %165 = ttir.empty() : tensor<8x1000xf32>
    %166 = "ttir.typecast"(%164, %165) <{conservative_folding = false}> : (tensor<8x1000xbf16>, tensor<8x1000xf32>) -> tensor<8x1000xf32>
    %167 = ttir.empty() : tensor<8xf32>
    %168 = "ttir.sum"(%166, %167) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<8x1000xf32>, tensor<8xf32>) -> tensor<8xf32>
    %169 = ttir.empty() : tensor<8x1xf32>
    %170 = "ttir.reshape"(%168, %169) <{shape = [8 : i32, 1 : i32]}> : (tensor<8xf32>, tensor<8x1xf32>) -> tensor<8x1xf32>
    %171 = ttir.empty() : tensor<8x1xf32>
    %172 = "ttir.broadcast"(%170, %171) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x1xf32>
    %173 = ttir.empty() : tensor<8x1xbf16>
    %174 = "ttir.typecast"(%172, %173) <{conservative_folding = false}> : (tensor<8x1xf32>, tensor<8x1xbf16>) -> tensor<8x1xbf16>
    %175 = ttir.empty() : tensor<8x1000xbf16>
    %176 = "ttir.broadcast"(%174, %175) <{broadcast_dimensions = array<i64: 1, 1000>}> : (tensor<8x1xbf16>, tensor<8x1000xbf16>) -> tensor<8x1000xbf16>
    %177 = ttir.empty() : tensor<8x1000xbf16>
    %178 = "ttir.div"(%164, %176, %177) : (tensor<8x1000xbf16>, tensor<8x1000xbf16>, tensor<8x1000xbf16>) -> tensor<8x1000xbf16>
    %179 = "ttir.mesh_shard"(%178) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<8x1000xbf16>) -> tensor<8x1000xbf16>
    return %179 : tensor<8x1000xbf16>
  }
}

