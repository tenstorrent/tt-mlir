// RUN: ttmlir-opt --ttcore-register-device --ttnn-allocate-distributed-op-semaphores -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttnn.buffer_type<l1>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_input_ws = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 128 + d3), <1x4, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>
#ttnn_layout_weight_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x32xbf16, #system_memory>>

// The pass walks DistributedOpInterface ops with unbound semaphore operands
// and inserts a ttnn.create_global_semaphore in the function prelude for
// each missing semaphore. The op must already have a memory_config attribute
// (the workaround pattern populates this), since the semaphore core range
// is derived from the input shard spec.
module @test_allocate_semaphores attributes {} {
  func.func public @test_allocates_global_semaphore(
      %arg0: tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>,
      %arg1: tensor<4x32xbf16, #ttnn_layout_weight_rm>) -> tensor<1x1x32x128xbf16, #ttnn_layout_input_ws> {
    // CHECK-LABEL: func.func public @test_allocates_global_semaphore
    // The semaphore must sit in the prelude alongside any other distributed-op
    // resources so trace hoisting can keep them all in one contiguous block.
    // CHECK: %[[DEV:.+]] = "ttnn.get_device"
    // CHECK: %[[SEM:.+]] = "ttnn.create_global_semaphore"
    // The op must consume the new semaphore; segment sizes update to
    // 1,1,0,0,1,1 (input, weight, residual, stats, semaphore, device).
    // CHECK: "ttnn.distributed_rms_norm"
    // CHECK-SAME: %[[SEM]]
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 0, 0, 1, 1>
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{
      cluster_axis = 1 : ui32,
      epsilon = 1.000000e-05 : f32,
      operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>,
      memory_config = #ttnn.memory_config<#l1, <width_sharded>, <shard_spec<grid = #ttnn.core_range_set<[<#ttnn.core_range<<0, 0>, <3, 0>>]>>, <32x32>, <row_major>, <physical>>>>
    }> : (tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>, tensor<4x32xbf16, #ttnn_layout_weight_rm>, !ttnn.device) -> tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>
    return %1 : tensor<1x1x32x128xbf16, #ttnn_layout_input_ws>
  }
}
