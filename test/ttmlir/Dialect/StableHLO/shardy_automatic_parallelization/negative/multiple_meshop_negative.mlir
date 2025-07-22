// REQUIRES: stablehlo
// RUN: not ttmlir-opt --automatic-sharding-pipeline="mesh-shape=1,2" %s 2>&1 | FileCheck %s

sdy.mesh @mesh = <["model"=1, "batch"=2]>
sdy.mesh @mesh2 = <["model"=1, "batch"=2]>

func.func @multiple_meshop_negative(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK: error: Shardy automatic parallelization pass only works on 1 meshOp for now
