// REQUIRES: opmodel
// RUN: ttmlir-opt \
// RUN:   --ttir-to-ttnn-backend-pipeline="optimization-level=2 mock-system-desc-arch=wormhole_b0 mesh-shape=1,2" \
// RUN:   -o %t %s
// RUN: FileCheck %s --input-file=%t

// At opt-level 2 the DistributedRMSNormWidthShardInput workaround is gated OFF; the greedy
// optimizer + DistributedRMSNormRuleBook own the layout, validated by
// OpModel<DistributedRMSNormOp> (which proxies to rms_norm). Verify the fused norm gets a
// width-sharded L1 input AND a generated LayerNormShardedMultiCoreProgramConfig — i.e. the
// optimizer/rulebook path produced both, not the hand-rolled workaround.

// CHECK-DAG: #[[L1_WS:.*]] = #ttnn.ttnn_layout<{{.*}}bf16{{.*}}#l1>{{.*}}<width_sharded>
// CHECK-LABEL: func.func @main
// CHECK: "ttnn.distributed_rms_norm"
// CHECK-SAME: program_config = #ttnn.layernorm_sharded_multicore_program_config
// CHECK-SAME: tensor<1x1x32x128xbf16, #[[L1_WS]]>

module @test_distributed_rms_norm_opt2 attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @main(
      %arg0: tensor<1x1x32x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>,
                                       ttcore.shard_status = #ttcore.shard_status<unsharded>},
      %arg1: tensor<128xbf16>        {ttcore.argument_type = #ttcore.argument_type<parameter>,
                                       ttcore.shard_status = #ttcore.shard_status<unsharded>}
  ) -> (tensor<1x1x32x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = "ttir.relu"(%arg0) : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    %1 = "ttir.distributed_rms_norm"(%0, %arg1) <{
        cluster_axis = 1 : ui32,
        epsilon = 1.000000e-05 : f32,
        operandSegmentSizes = array<i32: 1, 1, 0>
    }> : (tensor<1x1x32x128xbf16>, tensor<128xbf16>) -> tensor<1x1x32x128xbf16>
    return %1 : tensor<1x1x32x128xbf16>
  }
}
