// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,8" -o %t_explicit.mlir %s
// RUN: FileCheck %s --check-prefix=CHECK-EXPLICIT --input-file=%t_explicit.mlir

// CHECK-LABEL: module {
// CHECK-EXPLICIT-LABEL: module {
module {
  // When automatic-arg-analysis is disabled (default) and no mesh-shape is
  // provided, fall back to a 1x1 mesh.
  // CHECK: sdy.mesh @mesh = <["x"=1, "y"=1]>
  // When automatic-arg-analysis is disabled and a 2D mesh-shape is provided,
  // use it verbatim instead of falling back to a 1x1 mesh.
  // CHECK-EXPLICIT: sdy.mesh @mesh = <["x"=1, "y"=8]>
  func.func @single_chip(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
      %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
      %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
      return %2 : tensor<1x128xf32>
  }
}

// -----
