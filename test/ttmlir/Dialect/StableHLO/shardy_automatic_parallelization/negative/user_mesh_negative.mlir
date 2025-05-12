// REQUIRES: stablehlo
// RUN: not ttmlir-opt --automatic-sharding-pipeline="mesh-shape=2,4 automatic-arg-analysis" %s 2>&1 | FileCheck %s

func.func @user_mesh_negative(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK: error: Currently, shardy automatic parallel pass only supports 2d mesh shape and mesh shape dim0 must be 1
