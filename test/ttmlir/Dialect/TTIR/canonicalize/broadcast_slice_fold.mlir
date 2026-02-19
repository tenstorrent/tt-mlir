// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // Slice through add then through broadcast — the real use case.
  // The large 1x49920x6x5120 tensors should not appear after canonicalization.
  func.func @hoist_slice_through_add_and_broadcast(
      %a: tensor<1x1x6x5120xf32>,
      %b: tensor<1x1x6x5120xf32>) -> tensor<1x49920x1x5120xf32> {
    // CHECK-NOT: tensor<1x49920x6x5120xf32>
    // CHECK-DAG: "ttir.slice_static"(%arg0)
    // CHECK-DAG: "ttir.slice_static"(%arg1)
    // CHECK-DAG: "ttir.broadcast"
    // CHECK-DAG: "ttir.broadcast"
    // CHECK: "ttir.add"
    %0 = "ttir.broadcast"(%a) <{broadcast_dimensions = array<i64: 1, 49920, 1, 1>}> : (tensor<1x1x6x5120xf32>) -> tensor<1x49920x6x5120xf32>
    %1 = "ttir.broadcast"(%b) <{broadcast_dimensions = array<i64: 1, 49920, 1, 1>}> : (tensor<1x1x6x5120xf32>) -> tensor<1x49920x6x5120xf32>
    %2 = "ttir.add"(%0, %1) : (tensor<1x49920x6x5120xf32>, tensor<1x49920x6x5120xf32>) -> tensor<1x49920x6x5120xf32>
    %3 = "ttir.slice_static"(%2) <{begins = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ends = [1 : i32, 49920 : i32, 2 : i32, 5120 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x49920x6x5120xf32>) -> tensor<1x49920x1x5120xf32>
    return %3 : tensor<1x49920x1x5120xf32>
  }

  // Slice directly on broadcast (no eltwise).
  func.func @hoist_slice_above_broadcast(%arg0: tensor<1x1x6x5120xf32>) -> tensor<1x49920x1x5120xf32> {
    // CHECK-NOT: tensor<1x49920x6x5120xf32>
    // CHECK: "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 2 : i32, 5120 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
    // CHECK: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 49920, 1, 1>}> : (tensor<1x1x6x5120xf32>) -> tensor<1x49920x6x5120xf32>
    %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ends = [1 : i32, 49920 : i32, 2 : i32, 5120 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x49920x6x5120xf32>) -> tensor<1x49920x1x5120xf32>
    return %1 : tensor<1x49920x1x5120xf32>
  }

  // Broadcast has two uses (slice + direct return): the slice use is hoisted,
  // the original large broadcast is kept for the non-slice use.
  func.func @hoist_with_multiple_broadcast_uses(%arg0: tensor<1x1x6x5120xf32>)
      -> (tensor<1x49920x1x5120xf32>, tensor<1x49920x6x5120xf32>) {
    // Original large broadcast retained for the direct return.
    // CHECK: "ttir.broadcast"{{.*}}tensor<1x49920x6x5120xf32>
    // Slice is hoisted: operates on the small input, not the large broadcast.
    // CHECK: "ttir.slice_static"(%arg0)
    // CHECK-NOT: "ttir.slice_static"{{.*}}tensor<1x49920x6x5120xf32>
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 49920, 1, 1>}> : (tensor<1x1x6x5120xf32>) -> tensor<1x49920x6x5120xf32>
    %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ends = [1 : i32, 49920 : i32, 2 : i32, 5120 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x49920x6x5120xf32>) -> tensor<1x49920x1x5120xf32>
    return %1, %0 : tensor<1x49920x1x5120xf32>, tensor<1x49920x6x5120xf32>
  }
}
