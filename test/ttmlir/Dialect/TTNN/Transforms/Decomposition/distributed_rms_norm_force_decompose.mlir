// RUN: ttmlir-opt --ttcore-register-device --ttnn-decomposition -o %t %s
// RUN: FileCheck %s --check-prefix=FUSED --input-file=%t
// RUN: env TT_DISABLE_FUSED_RMSNORM=1 ttmlir-opt --ttcore-register-device --ttnn-decomposition -o %t2 %s
// RUN: FileCheck %s --check-prefix=DECOMP --input-file=%t2

// Debug toggle test (Kimi-K2.6 b128 decode isolation): the same eligible
// (1,1,32,896) op stays on the fused kernel by default, but decomposes into
// primitive ops when TT_DISABLE_FUSED_RMSNORM is set in the environment.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_supported = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 896 + d3), <1x1>, memref<1x28x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_weight = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x28x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module @test_distributed_rms_norm_force_decompose attributes {} {
  func.func public @test_force_decompose_toggle(
      %arg0: tensor<1x1x32x896xbf16, #ttnn_layout_supported>,
      %arg1: tensor<896xbf16, #ttnn_layout_weight>) -> tensor<1x1x32x896xbf16, #ttnn_layout_supported> {
    // FUSED-LABEL: func.func public @test_force_decompose_toggle
    // Default: the eligible (1,1,32,896) op stays on the fused kernel.
    // FUSED: "ttnn.distributed_rms_norm"
    // FUSED-SAME: tensor<1x1x32x896
    // FUSED-NOT: "ttnn.rsqrt"

    // DECOMP-LABEL: func.func public @test_force_decompose_toggle
    // With TT_DISABLE_FUSED_RMSNORM set: the fused op is gone and the op is
    // lowered through the decomposed path.
    // DECOMP-NOT: "ttnn.distributed_rms_norm"
    // DECOMP: "ttnn.rms_norm_pre_all_gather"
    // DECOMP: "ttnn.all_gather"
    // DECOMP: "ttnn.mean"
    // DECOMP: "ttnn.rsqrt"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>}> : (tensor<1x1x32x896xbf16, #ttnn_layout_supported>, tensor<896xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x1x32x896xbf16, #ttnn_layout_supported>
    return %1 : tensor<1x1x32x896xbf16, #ttnn_layout_supported>
  }
}
