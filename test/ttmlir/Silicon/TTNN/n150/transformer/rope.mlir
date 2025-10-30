// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module @SyncTensorsGraph.21 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func @main(%arg0: tensor<1x1024x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<1x32x1024x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg2: tensor<1x1024x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<1x32x1024x64xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = ttir.empty() : tensor<1x1x1024x64xbf16>
    %1 = "ttir.reshape"(%arg2, %0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x1024x64xbf16>, tensor<1x1x1024x64xbf16>) -> tensor<1x1x1024x64xbf16>
    %2 = ttir.empty() : tensor<1x32x1024x64xbf16>
    %3 = "ttir.broadcast"(%1, %2) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %4 = ttir.empty() : tensor<1x32x1024x64xbf16>
    %5 = "ttir.multiply"(%arg1, %3, %4) : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %6 = ttir.empty() : tensor<1x32x1024x32xbf16>
    %7 = "ttir.slice_static"(%arg1, %6) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x32xbf16>) -> tensor<1x32x1024x32xbf16>
    %8 = ttir.empty() : tensor<1x32x1024x32xbf16>
    %9 = "ttir.neg"(%7, %8) : (tensor<1x32x1024x32xbf16>, tensor<1x32x1024x32xbf16>) -> tensor<1x32x1024x32xbf16>
    %10 = ttir.empty() : tensor<1x32x1024x32xbf16>
    %11 = "ttir.slice_static"(%arg1, %10) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 1024 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x32xbf16>) -> tensor<1x32x1024x32xbf16>
    %12 = ttir.empty() : tensor<1x32x1024x64xbf16>
    %13 = "ttir.concat"(%9, %11, %12) <{dim = 3 : si32}> : (tensor<1x32x1024x32xbf16>, tensor<1x32x1024x32xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %14 = ttir.empty() : tensor<1x1x1024x64xbf16>
    %15 = "ttir.reshape"(%arg0, %14) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x1024x64xbf16>, tensor<1x1x1024x64xbf16>) -> tensor<1x1x1024x64xbf16>
    %16 = ttir.empty() : tensor<1x32x1024x64xbf16>
    %17 = "ttir.broadcast"(%15, %16) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %18 = ttir.empty() : tensor<1x32x1024x64xbf16>
    %19 = "ttir.multiply"(%13, %17, %18) : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %20 = ttir.empty() : tensor<1x32x1024x64xbf16>
    %21 = "ttir.add"(%5, %19, %20) : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    return %21 : tensor<1x32x1024x64xbf16>
  }
}
