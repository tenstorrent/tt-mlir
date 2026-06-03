// RUN: ttmlir-opt --ttcore-register-device --ttnn-allocate-distributed-op-buffers -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttnn.buffer_type<l1>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_input_ws = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 128 + d3), <1x4>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (3,0)>]>>
#ttnn_layout_weight_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x32xbf16, #system_memory>>

module @test_allocate_buffers attributes {} {
  func.func public @test_allocates_stats_buffer(
      %arg0: tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>,
      %arg1: tensor<4x32xbf16, #ttnn_layout_weight_rm>) -> tensor<1x1x32x128xbf16, #ttnn_layout_input_ws> {
    // CHECK: #[[STATS_LAYOUT:.+]] = #ttnn.ttnn_layout{{.*}}tile<32x32, f32>{{.*}}<width_sharded>
    // CHECK-LABEL: func.func public @test_allocates_stats_buffer
    // CHECK: %[[DEV:.+]] = "ttnn.get_device"
    // CHECK: %[[STATS:.+]] = "ttnn.empty"
    // CHECK-SAME: tensor<1x1x32x32xf32, #[[STATS_LAYOUT]]>
    // CHECK: "ttnn.distributed_rms_norm"
    // CHECK-SAME: %[[STATS]]
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 1>
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{
      cluster_axis = 1 : ui32,
      epsilon = 1.000000e-05 : f32,
      operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>,
      compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>
    }> : (tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>, tensor<4x32xbf16, #ttnn_layout_weight_rm>, !ttnn.device) -> tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>
    return %1 : tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>
  }
}
