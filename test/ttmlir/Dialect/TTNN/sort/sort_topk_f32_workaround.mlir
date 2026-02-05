// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s | FileCheck %s
// Test case from TopK operation that exposed the f32 input issue
// This is a regression test for the sort f32 input workaround

#loc1 = loc("p0.1")
module @SyncTensorsGraph.25 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.25 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @main(%arg0: tensor<1x10xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x10xf32>>, ttir.name = "args_0"} loc("p0.1")) -> (tensor<1x5xi64> {ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x5xi64>>}) {
        // CHECK-LABEL: @main
        // Verify f32→bf16 typecast is inserted before sort
        // CHECK: ttnn.typecast
        // CHECK: dtype = #ttcore.supportedDataTypes<bf16>
        // CHECK: tensor<1x10xf32
        // CHECK: tensor<1x10xbf16
        // Verify sort operates on bf16 tensors
        // CHECK: ttnn.sort
        // CHECK: descending = true
        // CHECK: tensor<1x10xbf16
        // CHECK: tensor<1x10xbf16
        // CHECK: tensor<1x10xui16
        %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 10 : i32]}> : (tensor<1x10xf32>) -> tensor<1x1x10xf32> loc(#loc2)
        %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 10 : i32]}> : (tensor<1x1x10xf32>) -> tensor<1x10xf32> loc(#loc3)
        %2 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 10 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<10xi32> loc(#loc4)
        %values, %indices = "ttir.sort"(%1) <{descending = true, dim = 1 : si32, stable = false}> : (tensor<1x10xf32>) -> (tensor<1x10xf32>, tensor<1x10xi32>) loc(#loc5)
        %3 = "ttir.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [1 : i32, 5 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1x10xi32>) -> tensor<1x5xi32> loc(#loc6)
        %4 = "ttir.typecast"(%3) <{conservative_folding = false}> : (tensor<1x5xi32>) -> tensor<1x5xi64> loc(#loc7)
        return %4 : tensor<1x5xi64> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("reshape.2")
#loc3 = loc("reshape.4")
#loc4 = loc("iota.5")
#loc5 = loc("sort.18")
#loc6 = loc("slice.22")
#loc7 = loc("convert.23")
