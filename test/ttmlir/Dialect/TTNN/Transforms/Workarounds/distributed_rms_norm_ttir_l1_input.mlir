// REQUIRES: opmodel
// RUN: ttmlir-opt \
// RUN:   --ttir-to-ttnn-backend-pipeline="optimization-level=2 mock-system-desc-arch=wormhole_b0 mesh-shape=1,2" \
// RUN:   -o %t %s
// RUN: FileCheck %s --input-file=%t

// TTIR-level test: starting from ttir.relu → ttir.distributed_rms_norm,
// verify that after the full TTIR→TTNN pipeline (workaround + greedy
// optimizer including L1SpillManagement) the relu output is converted to
// L1 width-sharded before reaching distributed_rms_norm with no
// intermediate DRAM spill on the input path.

// CHECK-LABEL: func.func @main
// relu → to_memory_config(L1 width-sharded) → distributed_rms_norm.
// Verify no DRAM spill appears before the L1 conversion and none between
// the L1 conversion and distributed_rms_norm (i.e. the input is never
// bounced through DRAM on its way to the fused kernel).
// CHECK: "ttnn.relu"
// CHECK-NOT: "ttnn.to_memory_config"{{.*}}#dram
// CHECK: "ttnn.to_memory_config"{{.*}}#l1{{.*}}width_sharded
// CHECK-NOT: "ttnn.to_memory_config"{{.*}}#dram
// CHECK: "ttnn.distributed_rms_norm"

module @test_ttir_rms_l1_input attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @main(
      %arg0: tensor<1x1x32x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>,
                                       ttcore.shard_status = #ttcore.shard_status<unsharded>},
      %arg1: tensor<128xbf16>        {ttcore.argument_type = #ttcore.argument_type<parameter>,
                                       ttcore.shard_status = #ttcore.shard_status<unsharded>}
  ) -> (tensor<1x1x32x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    // relu produces an L1 tensor that is tracked in liveValues by
    // L1SpillManagement. The workaround converts it to L1 width-sharded via a
    // ToLayoutOp (pre-decomposition) / to_memory_config (post-decomposition).
    %0 = "ttir.relu"(%arg0) : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    %1 = "ttir.distributed_rms_norm"(%0, %arg1) <{
        cluster_axis = 1 : ui32,
        epsilon = 1.000000e-05 : f32,
        operandSegmentSizes = array<i32: 1, 1, 0>
    }> : (tensor<1x1x32x128xbf16>, tensor<128xbf16>) -> tensor<1x1x32x128xbf16>
    return %1 : tensor<1x1x32x128xbf16>
  }
}
