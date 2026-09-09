// RUN: ttmlir-opt --ttcore-register-device --ttnn-allocate-distributed-op-semaphores -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttnn.buffer_type<l1>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_input_ws = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 128 + d3), <1x4>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (3,0)>]>>
#ttnn_layout_weight_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x32xbf16, #system_memory>>

module @test_allocate_semaphores attributes {} {
  func.func public @test_allocates_global_semaphore(
      %arg0: tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>,
      %arg1: tensor<4x32xbf16, #ttnn_layout_weight_rm>) -> tensor<1x1x32x128xbf16, #ttnn_layout_input_ws> {
    // CHECK-LABEL: func.func public @test_allocates_global_semaphore
    // CHECK: %[[DEV:.+]] = "ttnn.get_device"
    // CHECK: %[[SEM:.+]] = "ttnn.create_global_semaphore"
    // CHECK: "ttnn.distributed_rms_norm"
    // CHECK-SAME: %[[SEM]]
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 0, 0, 1, 1>
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{
      cluster_axis = 1 : ui32,
      epsilon = 1.000000e-05 : f32,
      operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>,
      memory_config = #ttnn.memory_config<#l1, <width_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (3,0)>]>, <32x32>, <row_major>>>
    }> : (tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>, tensor<4x32xbf16, #ttnn_layout_weight_rm>, !ttnn.device) -> tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>
    return %1 : tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>
  }
}
