// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% dst-allocation-strategy=legacy" %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% dst-allocation-strategy=graph-coloring-greedy" %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% dst-allocation-strategy=graph-coloring-cb" %s | FileCheck %s
//
// Tests that reduction operations (e.g., ttir.sum) work correctly with both legacy
// and graph-coloring DST allocation strategies. Reductions require special handling:
// 1. PackerMaskResetOp (reduce_uninit) must be inserted after tile_reduce ops
//    on all but the last iteration to reset accumulator state
// 2. Prologue copy from output CB to DST must be wrapped in a first-iteration
//    guard since the output CB contains uninitialized data on iteration 0

// CHECK-LABEL: func.func private @compute_kernel8

// First-iteration guard: skip copy_tile when any iter_index != 0
// CHECK: if %[[COND:.*]]
// CHECK-NEXT: call_opaque "copy_tile_init"
// CHECK-NEXT: call_opaque "copy_tile"

// Reduction operations followed by last-iteration guard
// CHECK: call_opaque "reduce_init"
// CHECK: call_opaque "reduce_tile"
// CHECK: if %
// CHECK-NEXT: call_opaque "reduce_uninit"

// CHECK: call_opaque "pack_tile"
// CHECK: call_opaque "tile_regs_release"

module {
  func.func @reductions_constrained_inputs(%arg0: tensor<256x256xf32>) -> tensor<1x1xf32> {
    %0 = ttir.empty() : tensor<1x1xf32>
    %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = true}> : (tensor<256x256xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    return %1 : tensor<1x1xf32>
  }
}
