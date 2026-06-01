// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2 enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that trace works correctly with presharded args.

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
    // Verify that capture trace creates valid local shape tensors.
    // CHECK-LABEL: func.func private @run_and_capture_trace_0_main
    // CHECK-SAME: %arg0: tensor<5x128x512xbf16,
    // CHECK-SAME: %arg1: tensor<5x128x512xbf16,
    // CHECK: "ttnn.empty"
    // CHECK-SAME: -> tensor<5x128x512xbf16,

    // CHECK-LABEL: func.func @main(
    func.func @main(%arg0: tensor<5x128x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>} loc("p0.2"), %arg1: tensor<5x128x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>} loc("p1.3")) -> (tensor<5x128x512xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>}) {
        %0 = "ttir.add"(%arg1, %arg0) : (tensor<5x128x512xbf16>, tensor<5x128x512xbf16>) -> tensor<5x128x512xbf16>
        return %0 : tensor<5x128x512xbf16>
    }
}
