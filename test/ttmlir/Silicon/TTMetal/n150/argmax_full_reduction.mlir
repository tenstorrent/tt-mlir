// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir

// Regression test for an intermittent use-after-free in D2MGridSelectionPass.
//
// A full-reduction argmax (reduce over all dims) decomposes into a long chain
// of d2m.generic ops where one producer (the row-reduced max) fans out to two
// separate to_layout->mask->generic consumers. GridSelection analyzed every
// generic's operands up front, then rewrote/erased producer ops per generic in
// a loop; an earlier generic's rewrite freed an op that a later generic's
// cached operand Value still pointed at, so getDefiningOp<ViewLayoutOp>() in
// GridAnalysis::isTTNNOperand dereferenced freed memory. The crash was
// nondeterministic (heap reuse + timing). Operands are now re-fetched live from
// the owning generic at apply time, so the pipeline must complete cleanly.
// CHECK-LABEL: func @argmax_full_reduction
func.func @argmax_full_reduction(%arg0: tensor<32x32xbf16>) -> tensor<1x1xi32> {
  // CHECK: ttmetal.enqueue_program
  %0 = "ttir.argmax"(%arg0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = true}> : (tensor<32x32xbf16>) -> tensor<1x1xi32>
  return %0 : tensor<1x1xi32>
}
