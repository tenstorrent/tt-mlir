// REQUIRES: stablehlo
// RUN: not ttmlir-opt --stablehlo-pipeline="mesh-shape=2,4" %s 2>&1 | FileCheck %s

func.func public @abs(%arg0: tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32> {
  %0 = stablehlo.abs %arg0 : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}

// CHECK: error: Could not find sdy, gspmd, tt annotations and automatic arg analysis is disabled. Skipping pass.
