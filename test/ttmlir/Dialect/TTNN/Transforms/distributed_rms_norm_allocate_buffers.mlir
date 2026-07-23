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
    // The stats buffer is always bf16 (matching production llama_ccl.py),
    // independent of the compute config's fp32_dest_acc_en = true below.
    // Match the stats layout specifically by its 2-tile-wide shard (last dim
    // 64 = 2*32 on a single core), not the input layout which is also bf16 +
    // width_sharded but 1 tile wide per core.
    // CHECK: #[[STATS_LAYOUT:.+]] = #ttnn.ttnn_layout{{.*}}memref<1x2x!ttcore.tile<32x32, bf16>, #l1>{{.*}}<width_sharded>
    // CHECK-LABEL: func.func public @test_allocates_stats_buffer
    // CHECK: %[[DEV:.+]] = "ttnn.get_device"
    // The stats buffer holds one 32-wide tile per device on the cluster axis
    // (mesh_shape 1x2, cluster_axis 1 -> 2 devices -> last dim 2*32 = 64), so
    // the fused kernel averages E(x^2) over all devices rather than a single
    // one (num_distributed_devices = padded_shape[-1] / 32). It is allocated
    // transiently -- right before the op -- and freed right after, so it does
    // not persist in L1. The unique transient id keeps CSE from merging the
    // per-op scratch buffers into one shared/persistent buffer.
    // CHECK: %[[STATS:.+]] = "ttnn.empty"
    // CHECK-SAME: tensor<1x1x32x64xbf16, #[[STATS_LAYOUT]]>
    // CHECK-SAME: ttnn.transient_stats_id
    // CHECK: "ttnn.distributed_rms_norm"
    // CHECK-SAME: %[[STATS]]
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 1>
    // CHECK: "ttnn.deallocate"(%[[STATS]])
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{
      cluster_axis = 1 : ui32,
      epsilon = 1.000000e-05 : f32,
      operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>,
      compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>
    }> : (tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>, tensor<4x32xbf16, #ttnn_layout_weight_rm>, !ttnn.device) -> tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>
    return %1 : tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>
  }

  // Locks the cluster_axis = 0 branch, which reads mesh_shape.getY(). MeshShapeAttr
  // stores (y, x); mesh_shape 4x1 -> y = 4 -> stats last dim 4*32 = 128 (4 tiles).
  func.func public @test_allocates_stats_buffer_cluster_axis_0(
      %arg0: tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>,
      %arg1: tensor<4x32xbf16, #ttnn_layout_weight_rm>) -> tensor<1x1x32x128xbf16, #ttnn_layout_input_ws> {
    // 4-tile-wide shard uniquely identifies the stats layout (input is 1 tile/core).
    // CHECK: #[[STATS_LAYOUT_Y:.+]] = #ttnn.ttnn_layout{{.*}}memref<1x4x!ttcore.tile<32x32, bf16>, #l1>{{.*}}<width_sharded>
    // CHECK-LABEL: func.func public @test_allocates_stats_buffer_cluster_axis_0
    // CHECK: %[[STATS_Y:.+]] = "ttnn.empty"
    // CHECK-SAME: tensor<1x1x32x128xbf16, #[[STATS_LAYOUT_Y]]>
    // CHECK: "ttnn.distributed_rms_norm"
    // CHECK-SAME: %[[STATS_Y]]
    // CHECK: "ttnn.deallocate"(%[[STATS_Y]])
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 4x1>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{
      cluster_axis = 0 : ui32,
      epsilon = 1.000000e-05 : f32,
      operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>,
      compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = false, packer_l1_acc = false>
    }> : (tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>, tensor<4x32xbf16, #ttnn_layout_weight_rm>, !ttnn.device) -> tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>
    return %1 : tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>
  }
}
