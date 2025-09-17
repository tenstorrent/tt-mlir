// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=false override-conv2d-config=conv2d_1=weights_dtype#bf16:activation#relu:deallocate_activation#false:reallocate_halo_output#true:act_block_h_override#0:act_block_w_div#1:reshard_if_not_optimal#false:override_sharding_config#false:transpose_shards#true:output_layout#row_major:enable_act_double_buffer#false:enable_weights_double_buffer#false" -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @forward(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x32x32x64xbf16> {
    %0 = ttir.empty() : tensor<1x32x32x64xbf16>
    // CHECK: "ttnn.conv2d"{{.*}} conv2d_config = #ttnn.conv2d_config<
    // CHECK-SAME: weights_dtype = bf16
    // CHECK-SAME: activation = <op_type = relu>
    // CHECK-SAME: deallocate_activation = false
    // CHECK-SAME: reallocate_halo_output = true
    // CHECK-SAME: act_block_h_override = 0
    // CHECK-SAME: act_block_w_div = 1
    // CHECK-SAME: reshard_if_not_optimal = false
    // CHECK-SAME: override_sharding_config = false
    // CHECK-SAME: transpose_shards = true
    // CHECK-SAME: output_layout = row_major
    // CHECK-SAME: enable_act_double_buffer = false
    // CHECK-SAME: enable_weights_double_buffer = false
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
