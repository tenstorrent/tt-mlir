// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_supported = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 128 + d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Weight in 2D form (4, 32) as produced by ttnn-decomposition prior to this pass.
#ttnn_layout_weight_2d = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// The workaround pattern only converts layout / sharding. It must:
//   - Width-shard the input.
//   - Convert the weight from Tile to RowMajor.
//   - Leave stats / semaphore unbound (allocated by dedicated passes later).
module @test_distributed_rms_norm_workaround attributes {} {
  func.func public @test_workaround_layout_only(
      %arg0: tensor<1x1x32x128xbf16, #ttnn_layout_supported>,
      %arg1: tensor<4x32xbf16, #ttnn_layout_weight_2d>) -> tensor<1x1x32x128xbf16, #ttnn_layout_supported> {
    // CHECK-LABEL: func.func public @test_workaround_layout_only
    // Input is converted to a width-sharded L1 layout.
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: <l1>
    // CHECK-SAME: <width_sharded>
    // Weight is converted from Tile to RowMajor.
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: row_major
    // The fused op survives.
    // CHECK: "ttnn.distributed_rms_norm"
    // The workaround pass does not allocate stats or semaphores; both
    // operand segments stay zero.
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>
    // CHECK-NOT: "ttnn.empty"
    // CHECK-NOT: "ttnn.create_global_semaphore"
    // CHECK-NOT: "ttnn.rsqrt"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>}> : (tensor<1x1x32x128xbf16, #ttnn_layout_supported>, tensor<4x32xbf16, #ttnn_layout_weight_2d>, !ttnn.device) -> tensor<1x1x32x128xbf16, #ttnn_layout_supported>
    return %1 : tensor<1x1x32x128xbf16, #ttnn_layout_supported>
  }
}
