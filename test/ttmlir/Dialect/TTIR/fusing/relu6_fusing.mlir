// RUN: ttmlir-opt --canonicalize --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module @SyncTensorsGraph.2210 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
    func.func @main(%arg0: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "weight"} loc("xla__device_data"), %arg1: tensor<8x3x224x224xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "input"} loc("xla__device_data")) -> (tensor<8x32x112x112xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        // CHECK: "ttir.relu6"
        // CHECK-NOT: "ttir.relu"
        %0 = "ttir.constant"() <{value = dense<6.000000e+00> : tensor<8x32x112x112xbf16>}> : () -> tensor<8x32x112x112xbf16>
        %1 = "ttir.conv2d"(%arg1, %arg0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<8x3x224x224xbf16>, tensor<32x3x3x3xbf16>) -> tensor<8x32x112x112xbf16>
        %2 = "ttir.relu"(%1) : (tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16>
        %3 = "ttir.minimum"(%2, %0) : (tensor<8x32x112x112xbf16>, tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16>
        return %3 : tensor<8x32x112x112xbf16>
    }

    // Test: scalar full -> reshape pattern (produced by broadcast_in_dim conversion of scalar constants)
    func.func @relu6_scalar_reshape(%arg0: tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16> {
        // CHECK: "ttir.relu6"
        // CHECK-NOT: "ttir.maximum"
        // CHECK-NOT: "ttir.minimum"
        %0 = "ttir.full"() <{fill_value = 0.000000e+00 : f32, shape = array<i32>}> : () -> tensor<bf16>
        %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
        %2 = "ttir.maximum"(%arg0, %1) : (tensor<8x32x112x112xbf16>, tensor<1x1x1x1xbf16>) -> tensor<8x32x112x112xbf16>
        %3 = "ttir.full"() <{fill_value = 6.000000e+00 : f32, shape = array<i32>}> : () -> tensor<bf16>
        %4 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
        %5 = "ttir.minimum"(%2, %4) : (tensor<8x32x112x112xbf16>, tensor<1x1x1x1xbf16>) -> tensor<8x32x112x112xbf16>
        return %5 : tensor<8x32x112x112xbf16>
    }
}
