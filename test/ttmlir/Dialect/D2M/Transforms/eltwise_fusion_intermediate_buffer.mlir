// RUN: ttmlir-opt --ttir-to-ttmetal-fe-pipeline --d2m-elementwise-fusion %s | FileCheck %s

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

// CHECK-LABEL: func.func @three_op_chain
//
// Inside the fused region, capture the outs SSA value of the first
// linalg.generic. None of the subsequent linalg.generic ops in the IR may
// reuse that exact SSA value: each must have its own distinct intermediate
// (or final) output buffer.
// CHECK:     linalg.generic{{.*}}outs(%[[FIRST_OUT:[^ ]+]] : tensor
// CHECK-NOT: linalg.generic{{.*}}outs(%[[FIRST_OUT]] : tensor
