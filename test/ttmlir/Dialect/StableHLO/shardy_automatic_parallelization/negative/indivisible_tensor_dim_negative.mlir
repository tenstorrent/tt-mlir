// REQUIRES: stablehlo
// RUN: not ttmlir-opt --automatic-sharding-pipeline="mesh-shape=1,2" %s 2>&1 | FileCheck %s

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func public @indivisible_tensor_dim_negative(%arg0: tensor<13x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}, {}, {}]>}, %arg1: tensor<13x48x24x32xf32>) -> tensor<13x48x24x32xf32> {
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<13x48x24x32xf32>
  return %0 : tensor<13x48x24x32xf32>
}

// CHECK: error: Could not apply propagated tensor shardings to tensor dimensions
