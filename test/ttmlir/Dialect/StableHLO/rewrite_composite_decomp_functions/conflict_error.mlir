// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --verify-diagnostics --rewrite-composite-decomp-functions %s

// Two "tenstorrent.gather" composites share the same decomposition function
// but carry different `composite_attributes`. The rewriter can only install
// one body, so the pass must error out on the second (conflicting) op.
module @ConflictingAttrsSameDecomp {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    // expected-note @+1 {{prior composite referencing the same decomposition}}
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather.impl_conflict} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    // expected-error @+1 {{shares decomposition target '"tenstorrent.gather.impl_conflict"' with a prior composite but carries different composite_attributes}}
    %1 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 1 : i64}, decomposition = @tenstorrent.gather.impl_conflict} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @tenstorrent.gather.impl_conflict(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// The same conflict is detected when one of the composites has no
// `composite_attributes` at all (treated as empty dictionary) while the
// other carries a non-empty one.
module @ConflictPresentVsMissingAttrs {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    // expected-note @+1 {{prior composite referencing the same decomposition}}
    %0 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {decomposition = @tenstorrent.gather.impl_conflict2} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    // expected-error @+1 {{shares decomposition target '"tenstorrent.gather.impl_conflict2"' with a prior composite but carries different composite_attributes}}
    %1 = stablehlo.composite "tenstorrent.gather" %arg0, %arg1 {composite_attributes = {dim = 1 : i64}, decomposition = @tenstorrent.gather.impl_conflict2} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @tenstorrent.gather.impl_conflict2(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
