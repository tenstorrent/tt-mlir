// RUN: ttmlir-opt --ttir-to-emitc-pipeline="try-recover-structure=true system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
//
// This IR was generated using tt-xla's example found in
// https://github.com/tenstorrent/tt-xla/tree/main/examples/pytorch/codegen
//
// Check that structure recovery creates functions based on source locations.
//
// CHECK: func.func private @FirstModule
// CHECK: func.func private @SecondModule
// CHECK: func.func private @forward
// CHECK: func.func @main

#loc1 = loc("-1|unknown|unknown|-1|xla__device_data")
module @SyncTensorsGraph.17 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.17 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @main(%arg0: tensor<64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___m2_w"} loc("-1|unknown|unknown|-1|xla__device_data"), %arg1: tensor<32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___m1_w"} loc("-1|unknown|unknown|-1|xla__device_data"), %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("-1|unknown|unknown|-1|xla__device_data")) -> (tensor<32x128xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 32 : i32, 32 : i32]}> : (tensor<32x32xbf16>) -> tensor<1x32x32xbf16> loc(#loc2)
        %1 = "ttir.reshape"(%0) <{shape = [32 : i32, 32 : i32]}> : (tensor<1x32x32xbf16>) -> tensor<32x32xbf16> loc(#loc2)
        %2 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 32 : i32, 64 : i32]}> : (tensor<32x64xbf16>) -> tensor<1x32x64xbf16> loc(#loc2)
        %3 = "ttir.reshape"(%2) <{shape = [32 : i32, 64 : i32]}> : (tensor<1x32x64xbf16>) -> tensor<32x64xbf16> loc(#loc2)
        %4 = "ttir.dot_general"(%1, %3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x32xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16> loc(#loc3)
        %5 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<64x128xbf16>) -> tensor<1x64x128xbf16> loc(#loc2)
        %6 = "ttir.reshape"(%5) <{shape = [64 : i32, 128 : i32]}> : (tensor<1x64x128xbf16>) -> tensor<64x128xbf16> loc(#loc2)
        %7 = "ttir.dot_general"(%4, %6) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x64xbf16>, tensor<64x128xbf16>) -> tensor<32x128xbf16> loc(#loc4)
        %8 = "ttir.multiply"(%7, %7) : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16> loc(#loc5)
        return %8 : tensor<32x128xbf16> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("-1|unknown|unknown|-1|aten__view")
#loc3 = loc("0|FirstModule[m1]|/repos/tt-xla/examples/pytorch/codegen/prettify_example.py:27|forward|28|aten__mm")
#loc4 = loc("1|SecondModule[m2]|/repos/tt-xla/examples/pytorch/codegen/prettify_example.py:36|forward|37|aten__mm")
#loc5 = loc("2|/repos/tt-xla/examples/pytorch/codegen/prettify_example.py:46|forward|52|aten__mul")
