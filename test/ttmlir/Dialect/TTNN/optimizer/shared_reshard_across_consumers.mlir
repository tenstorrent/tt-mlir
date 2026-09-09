// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s --mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// Consumers that ask the same producer for the same target layout share one
// reshard op rather than getting one each. The canonical shape is the q/k/v
// slices off a fused-QKV matmul: three consumers, one producer, and a single
// layout requirement between them.
//
// The slices carry different offsets so CSE cannot collapse them, which is what
// makes the operand of each one meaningful: all three must read the *same*
// reshard result. Without the dedup this lowers to three separate
// ttnn.to_memory_config ops, each consuming the matmul directly.

module {
  func.func @fused_qkv_slices(
      %a: tensor<32x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %w: tensor<512x384xbf16> {ttcore.argument_type = #ttcore.argument_type<input>})
      -> tensor<32x128xbf16> {
    // CHECK-LABEL: func.func @fused_qkv_slices
    // CHECK: %[[QKV:[0-9]+]] = "ttnn.matmul"
    // CHECK: %[[RESHARD:[0-9]+]] = "ttnn.to_memory_config"(%[[QKV]])
    // CHECK: "ttnn.slice_static"(%[[RESHARD]])
    // CHECK: "ttnn.slice_static"(%[[RESHARD]])
    // CHECK: "ttnn.slice_static"(%[[RESHARD]])
    %0 = "ttir.matmul"(%a, %w) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
    %q = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 128 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x384xbf16>) -> tensor<32x128xbf16>
    %k = "ttir.slice_static"(%0) <{begins = [0 : i32, 128 : i32], ends = [32 : i32, 256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x384xbf16>) -> tensor<32x128xbf16>
    %v = "ttir.slice_static"(%0) <{begins = [0 : i32, 256 : i32], ends = [32 : i32, 384 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x384xbf16>) -> tensor<32x128xbf16>
    %1 = "ttir.add"(%q, %k) : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    %2 = "ttir.add"(%1, %v) : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    return %2 : tensor<32x128xbf16>
  }
}
