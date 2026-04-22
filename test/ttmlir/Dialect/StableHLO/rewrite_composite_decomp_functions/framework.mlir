// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --rewrite-composite-decomp-functions -o %t %s
// RUN: FileCheck %s --input-file=%t

// A composite op whose name is NOT in the rewriter registry is left
// completely untouched: the composite op stays, and its decomposition
// function body stays as-is (unchanged constant placeholder).
module @UnregisteredCompositeIsUntouched {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK: stablehlo.composite "some.other.composite"
    %0 = stablehlo.composite "some.other.composite" %arg0 {decomposition = @some.other.impl} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
  // CHECK-LABEL: func.func private @some.other.impl
  // CHECK-NEXT:  stablehlo.constant dense<4.200000e+01>
  // CHECK-NOT:   stablehlo.gather
  // CHECK-NOT:   stablehlo.reshape
  func.func private @some.other.impl(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = stablehlo.constant dense<4.200000e+01> : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// -----

// Registered and unregistered composites coexist: only the registered one's
// decomposition function is rewritten; the other is left alone.
module @MixedRegisteredAndUnregistered {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<5x3xf32>) {
    %0 = stablehlo.composite "tenstorrent.gather_dim" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather_dim.impl_mixed} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    %1 = stablehlo.composite "tt.unused" %arg0 {decomposition = @tt.unused.impl} : (tensor<5x3xf32>) -> tensor<5x3xf32>
    return %0, %1 : tensor<2x3xf32>, tensor<5x3xf32>
  }
  // The gather decomposition is rewritten.
  // CHECK-LABEL: func.func private @tenstorrent.gather_dim.impl_mixed
  // CHECK:       stablehlo.reshape
  // CHECK:       "stablehlo.gather"
  func.func private @tenstorrent.gather_dim.impl_mixed(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  // The unregistered composite's decomposition stays untouched.
  // CHECK-LABEL: func.func private @tt.unused.impl
  // CHECK-NEXT:  stablehlo.constant dense<7.000000e+00>
  // CHECK-NOT:   stablehlo.gather
  func.func private @tt.unused.impl(%arg0: tensor<5x3xf32>) -> tensor<5x3xf32> {
    %0 = stablehlo.constant dense<7.000000e+00> : tensor<5x3xf32>
    return %0 : tensor<5x3xf32>
  }
}

// -----

// Multiple composite ops that share the SAME decomposition function and
// carry IDENTICAL `composite_attributes` must rewrite that function exactly
// once and must not error out.
module @SharedDecompSameAttrs {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = stablehlo.composite "tenstorrent.gather_dim" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather_dim.impl_shared} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    %1 = stablehlo.composite "tenstorrent.gather_dim" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather_dim.impl_shared} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
  }
  // The shared decomposition function should contain exactly one reshape
  // and one gather — not two of each from duplicate rewrites.
  // CHECK-LABEL: func.func private @tenstorrent.gather_dim.impl_shared
  // CHECK-COUNT-1: stablehlo.reshape
  // CHECK-COUNT-1: stablehlo.gather
  // CHECK:         return
  func.func private @tenstorrent.gather_dim.impl_shared(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Two registered composites with DIFFERENT decomposition functions both get
// rewritten independently. Confirms the walk visits every matching
// composite, not just the first one.
module @TwoDistinctDecomps {
  func.func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = stablehlo.composite "tenstorrent.gather_dim" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather_dim.impl_a} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    %1 = stablehlo.composite "tenstorrent.gather_dim" %arg0, %arg1 {composite_attributes = {dim = 0 : i64}, decomposition = @tenstorrent.gather_dim.impl_b} : (tensor<5x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
    return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather_dim.impl_a
  // CHECK:       "stablehlo.gather"
  func.func private @tenstorrent.gather_dim.impl_a(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func private @tenstorrent.gather_dim.impl_b
  // CHECK:       "stablehlo.gather"
  func.func private @tenstorrent.gather_dim.impl_b(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
