// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 compute-cfg-math-fidelity=lofi compute-cfg-fp32-dest-acc-en=true" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#loc1 = loc("p0.1")
#loc2 = loc("p1.5")
#loc3 = loc("p2.7")
module @SyncTensorsGraph.14 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.14 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @main(%arg0: tensor<3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___conv_bias"} loc("p0.1"), %arg1: tensor<3x3x2x2xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___conv_weight"} loc("p1.5"), %arg2: tensor<1x3x224x224xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p2.7")) -> (tensor<1x3x223x223xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = "ttir.convolution"(%arg2, %arg1) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x3x224x224xbf16>, tensor<3x3x2x2xbf16>) -> tensor<1x3x223x223xbf16> loc(#loc4)
        %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 3 : i32]}> : (tensor<3xbf16>) -> tensor<1x1x3xbf16> loc(#loc5)
        %2 = "ttir.reshape"(%1) <{shape = [3 : i32]}> : (tensor<1x1x3xbf16>) -> tensor<3xbf16> loc(#loc6)
        %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 3 : i32, 1 : i32, 1 : i32]}> : (tensor<3xbf16>) -> tensor<1x3x1x1xbf16> loc(#loc7)
        %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 223, 223>}> : (tensor<1x3x1x1xbf16>) -> tensor<1x3x223x223xbf16> loc(#loc7)
        %5 = "ttir.add"(%0, %4) : (tensor<1x3x223x223xbf16>, tensor<1x3x223x223xbf16>) -> tensor<1x3x223x223xbf16> loc(#loc8)
        return %5 : tensor<1x3x223x223xbf16> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc4 = loc("convolution.9")
#loc5 = loc("reshape.2")
#loc6 = loc("reshape.4")
#loc7 = loc("transpose.11")
#loc8 = loc("add.12")
