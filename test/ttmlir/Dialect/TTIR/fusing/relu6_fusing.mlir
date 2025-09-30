module @SyncTensorsGraph.2210 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
    func.func @main(%arg0: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "weight"} loc("xla__device_data"), %arg1: tensor<8x3x224x224xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "input"} loc("xla__device_data"), %arg2: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "bn_scale"}, %arg3: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "bn_offset"}, %arg4: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "bn_mean"}, %arg5: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "bn_variance"}) -> (tensor<8x32x112x112xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = "ttir.constant"() <{value = dense<6.000000e+00> : tensor<8x32x112x112xbf16>}> : () -> tensor<8x32x112x112xbf16> 
        %1 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<8x32x112x112xbf16>}> : () -> tensor<8x32x112x112xbf16> 
        
        %2 = ttir.empty() : tensor<8x32x112x112xbf16> 
        %3 = "ttir.convolution"(%arg1, %arg0, %2) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<8x3x224x224xbf16>, tensor<32x3x3x3xbf16>, tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16> 
        
        %4 = ttir.empty() : tensor<8x32x112x112xbf16>
        %5 = "ttir.batch_norm"(%3, %arg2, %arg3, %arg4, %arg5, %4) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, training = false}> : (tensor<8x32x112x112xbf16>, tensor<32xbf16>, tensor<32xbf16>, tensor<32xbf16>, tensor<32xbf16>, tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16>
        
        %6 = ttir.empty() : tensor<8x32x112x112xbf16> 
        %7 = "ttir.maximum"(%5, %1, %6) : (tensor<8x32x112x112xbf16>, tensor<8x32x112x112xbf16>, tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16> 
        %8 = ttir.empty() : tensor<8x32x112x112xbf16> 
        %9 = "ttir.minimum"(%7, %0, %8) : (tensor<8x32x112x112xbf16>, tensor<8x32x112x112xbf16>, tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16> 
        return %9 : tensor<8x32x112x112xbf16> 
    }
}