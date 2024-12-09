// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @select_identity(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    // CHECK: %{{[0-9]+}} = "ttir.slice"
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 0: si32, length = 4: si32, stride = 4: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  func.func @select_multi_slice(%arg0: tensor<4x2x64x128xf32>) -> tensor<4x2x64x32xf32> {
    %0 = tensor.empty() : tensor<4x2x64x32xf32>

    // CHECK: %{{[0-9]+}} = "ttir.slice"
    // CHECK: %{{[0-9]+}} = "ttir.slice"
    // CHECK: %{{[0-9]+}} = "ttir.slice"
    // CHECK: %{{[0-9]+}} = "ttir.slice"
    // CHECK: %{{[0-9]+}} = "ttir.concat"
    %1 = "ttir.select"(%arg0, %0) <{dim = -1: si32, begin = 0: si32, length = 4: si32, stride = 16: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x2x64x128xf32>, tensor<4x2x64x32xf32>) -> tensor<4x2x64x32xf32>

    return %1 : tensor<4x2x64x32xf32>
  }
}
