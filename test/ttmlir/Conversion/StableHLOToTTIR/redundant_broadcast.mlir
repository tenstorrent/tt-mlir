// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<1xf32>) -> tensor<8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<8xf32>) -> tensor<8xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xf32>) -> tensor<8xf32>
    %2 = stablehlo.add %0, %1 : tensor<8xf32>
    // CHECK-NOT: %{{[0-9]+}} = "ttir.broadcast"(%arg0, %{{[0-9]+}}) <{dimension = [0], operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %2 : tensor<8xf32>
  }
}
