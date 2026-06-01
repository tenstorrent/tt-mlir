// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2 enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that mergeToLayoutOpsWithFuncArgs correctly handles the
// arg -> MeshShardOp -> ToLayoutOp pattern. After merging:
// - The function argument types should be updated with the target layout.
// - The MeshShardOp result type should be updated to match.
// - The ToLayoutOp should be removed.

module {
  // Verify trace function receives correctly typed tensors.
  // CHECK-LABEL: func.func private @trace_0_merge_to_layout_mesh_shard
  // CHECK: "ttnn.add"

  // CHECK-LABEL: func.func private @run_and_capture_trace_0_merge_to_layout_mesh_shard
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.deallocate"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"
  // CHECK: "ttnn.execute_trace"

  // CHECK-LABEL: func.func private @execute_trace_0_merge_to_layout_mesh_shard
  // CHECK: "ttnn.execute_trace"

  // CHECK-LABEL: func.func @merge_to_layout_mesh_shard(
  func.func @merge_to_layout_mesh_shard(%arg0: tensor<10x128x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>} loc("p0.2"), %arg1: tensor<10x128x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>} loc("p1.3")) -> (tensor<10x128x512xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<10x128x512xbf16>>}) {
    // Verify mesh_shard ops are present but no to_layout ops between
    // mesh_shard and capture_or_execute_trace. The mergeToLayoutOpsWithFuncArgs
    // function should merge the ToLayoutOps into function argument types and
    // update the mesh_shard result types accordingly.
    // CHECK: "ttnn.mesh_shard"
    // CHECK: "ttnn.mesh_shard"
    // CHECK-NOT: "ttnn.to_layout"
    // CHECK: "ttnn.capture_or_execute_trace"
    // CHECK-NOT: "ttnn.to_layout"
    // CHECK: return
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<10x128x512xbf16>) -> tensor<5x128x512xbf16>
    %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<10x128x512xbf16>) -> tensor<5x128x512xbf16>
    %2 = "ttir.add"(%0, %1) : (tensor<5x128x512xbf16>, tensor<5x128x512xbf16>) -> tensor<5x128x512xbf16>
    %3 = "ttir.mesh_shard"(%2) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<5x128x512xbf16>) -> tensor<10x128x512xbf16>
    return %3 : tensor<10x128x512xbf16>
  }
}
