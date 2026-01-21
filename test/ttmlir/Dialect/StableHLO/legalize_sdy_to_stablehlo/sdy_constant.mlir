// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

func.func @SdyConstant() -> tensor<f32> {
    // CHECK: stablehlo.constant
    %0 = sdy.constant dense<0.0> : tensor<f32>
    return %0 : tensor<f32>
}
