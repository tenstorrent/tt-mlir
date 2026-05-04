// RUN: ttmlir-opt --ttir-to-ttmetal-fe-pipeline --d2m-elementwise-fusion %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-fe-pipeline --d2m-elementwise-fusion %s | FileCheck %s --check-prefix=UNIQUE

// Regression test for the read-after-write hazard in fused eltwise generics
// when chaining 3+ elementwise ops. Previously, ElementwiseFusion mapped the
// producer's output tensor.empty to the consumer's first output tensor.empty,
// which meant every linalg.generic in the fused body wrote to the same buffer.
// Downstream that buffer became a single output circular buffer; the producer's
// pack_tile would write into it while a downstream linalg op in the same fused
// region was still reading stale tiles from the previous outer-loop iteration,
// corrupting the result.
//
// The fix creates a fresh intermediate tensor.empty for each producer's output
// inside the fused region, so each linalg.generic in the fused body has its
// own distinct outs buffer.

module {
  func.func @three_op_chain(
      %arg0: tensor<32x32xbf16>,
      %arg1: tensor<32x32xbf16>,
      %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.multiply"(%0, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.sigmoid"(%1) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }
}

// (1) Structural check: the chain fuses into a single compute d2m.generic,
//     and the fused region contains exactly three linalg.generic ops in the
//     original chain order: tile_add -> tile_mul -> tile_sigmoid. Any extra
//     or missing linalg.generic would trip these checks.
// CHECK-LABEL: func.func @three_op_chain
// CHECK:           linalg.generic
// CHECK:           "d2m.tile_add"
// CHECK:           linalg.generic
// CHECK:           "d2m.tile_mul"
// CHECK:           linalg.generic
// CHECK:           "d2m.tile_sigmoid"
// CHECK-NOT:       linalg.generic
// CHECK-NOT:       "d2m.tile_add"
// CHECK-NOT:       "d2m.tile_mul"
// CHECK-NOT:       "d2m.tile_sigmoid"

// (2) Aliasing check (separate FileCheck pass): each linalg.generic in the
//     fused region must have its OWN outs tensor.empty. Capture the first
//     linalg.generic's outs SSA value and assert no later linalg.generic
//     reuses it. Without the fix, all three linalgs share the same outs.
// UNIQUE-LABEL: func.func @three_op_chain
// UNIQUE:       linalg.generic{{.*}}outs(%[[FIRST_OUT:[^ ]+]] : tensor
// UNIQUE-NOT:   linalg.generic{{.*}}outs(%[[FIRST_OUT]] : tensor
