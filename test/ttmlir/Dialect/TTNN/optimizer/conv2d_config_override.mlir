// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true override-conv2d-config=conv2d_1=dtype#bf16:weights_dtype#bf16:activation#relu:input_channels_alignment#32:deallocate_activation#false:reallocate_halo_output#true:act_block_h_override#0:act_block_w_div#1:reshard_if_not_optimal#false:override_sharding_config#false:transpose_shards#true:output_layout#row_major:enable_act_double_buffer#false:enable_weights_double_buffer#false:enable_split_reader#false:enable_subblock_padding#false" %s | FileCheck %s
module {
  func.func @forward(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x32x32x64xbf16> {
    %0 = ttir.empty() : tensor<1x32x32x64xbf16>
    // CHECK: %{{.*}} = "ttnn.conv2d"{{.*}} conv2d_config = #ttnn.conv2d_config<dtype = bf16, weights_dtype = bf16, activation = "relu", input_channels_alignment = 32, deallocate_activation = 0, reallocate_halo_output = 1, act_block_h_override = 0, act_block_w_div = 1, reshard_if_not_optimal = 0, override_sharding_config = 0, transpose_shards = 1, output_layout = row_major, preprocess_weights_on_device = 0, always_preprocess_weights = 0, enable_act_double_buffer = 0, enable_weights_double_buffer = 0, enable_split_reader = 0, enable_subblock_padding = 0>{{.*}}
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = array<i32: 1, 1>,
              padding = array<i32: 1, 1>,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16> loc(#loc2)
    return %1 : tensor<1x32x32x64xbf16>
  }
}
#loc2 = loc("conv2d_1")
