// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2 enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that mesh shard ops aren't hoisted into the trace functions.

module {
    // Verify that capture trace creates valid local shape tensors.
    // CHECK-LABEL: func.func private @run_and_capture_trace_0_main
    // CHECK-SAME: %arg0: tensor<5x128x512xbf16,
    // CHECK-SAME: %arg1: tensor<5x128x512xbf16,
    // CHECK: "ttnn.empty"
    // CHECK-SAME: -> tensor<5x128x512xbf16,
    // CHECK: "ttnn.empty"
    // CHECK-SAME: -> tensor<5x128x512xbf16,

    // CHECK-LABEL: func.func @main(
    // CHECK: "ttnn.mesh_shard"
    // CHECK: "ttnn.mesh_shard"
    func.func @main(%arg0: tensor<10x128x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>} loc("p0.2"), %arg1: tensor<10x128x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>} loc("p1.3")) -> (tensor<10x128x512xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<10x128x512xbf16>>}) {
        %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<10x128x512xbf16>) -> tensor<5x128x512xbf16>
        %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<10x128x512xbf16>) -> tensor<5x128x512xbf16>
        %2 = "ttir.add"(%1, %0) : (tensor<5x128x512xbf16>, tensor<5x128x512xbf16>) -> tensor<5x128x512xbf16>
        %3 = "ttir.mesh_shard"(%2) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<5x128x512xbf16>) -> tensor<10x128x512xbf16>
        return %3 : tensor<10x128x512xbf16>
    }
}
