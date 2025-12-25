// RUN: ttmlir-opt --canonicalize --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module @SyncTensorsGraph.2210 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
    func.func @main(%arg0: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "weight"} loc("xla__device_data"), %arg1: tensor<8x3x224x224xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "input"} loc("xla__device_data")) -> (tensor<8x32x112x112xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        // CHECK: "ttir.relu6"
        // CHECK-NOT: "ttir.relu"
        %0 = "ttir.constant"() <{value = dense<6.000000e+00> : tensor<8x32x112x112xbf16>}> : () -> tensor<8x32x112x112xbf16>
        %1 = "ttir.permute"(%arg1) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<8x3x224x224xbf16>) -> tensor<8x224x224x3xbf16>
        %2 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<32x3x3x3xbf16>) -> tensor<32x3x3x3xbf16>
        %3 = "ttir.conv2d"(%1, %2) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> : (tensor<8x224x224x3xbf16>, tensor<32x3x3x3xbf16>) -> tensor<8x112x112x32xbf16>
        %4 = "ttir.permute"(%3) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<8x112x112x32xbf16>) -> tensor<8x32x112x112xbf16>
        %5 = "ttir.relu"(%4) : (tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16>
        %6 = "ttir.minimum"(%5, %0) : (tensor<8x32x112x112xbf16>, tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16>
        return %6 : tensor<8x32x112x112xbf16>
    }
}
