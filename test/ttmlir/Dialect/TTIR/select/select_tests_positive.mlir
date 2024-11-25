// RUN: ttmlir-opt %s | FileCheck %s

#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @select_identity(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    // CHECK: %{{[0-9]+}} = "ttir.select"
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 0: si32, length = 4: si32, stride = 4: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  func.func @select_half(%arg0: tensor<4x4xf32>) -> tensor<4x2xf32> {
    %0 = tensor.empty() : tensor<4x2xf32>
    // CHECK: %{{[0-9]+}} = "ttir.select"
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 0: si32, length = 2: si32, stride = 4: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
    return %1 : tensor<4x2xf32>
  }

  func.func @select_single(%arg0: tensor<4x4xf32>) -> tensor<4x1xf32> {
    %0 = tensor.empty() : tensor<4x1xf32>
    // CHECK: %{{[0-9]+}} = "ttir.select"
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 3: si32, length = 1: si32, stride = 1: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    return %1 : tensor<4x1xf32>
  }

  func.func @select_half_2_no_stride(%arg0: tensor<4x4xf32>) -> tensor<4x2xf32> {
    %0 = tensor.empty() : tensor<4x2xf32>
    // CHECK: %{{[0-9]+}} = "ttir.select"
    %1 = "ttir.select"(%arg0, %0) <{dim = 1: si32, begin = 2: si32, length = 2: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
    return %1 : tensor<4x2xf32>
  }

  func.func @select_neg_dim(%arg0: tensor<10x3x128x64xf32>) -> tensor<10x3x8x64xf32> {
    %0 = tensor.empty() : tensor<10x3x8x64xf32>
    // CHECK: %{{[0-9]+}} = "ttir.select"
    %1 = "ttir.select"(%arg0, %0) <{dim = -2: si32, begin = 0: si32, length = 2: si32, stride = 32: si32, operand_constraints = [#any_device_tile, #any_device_tile]}>  :
        (tensor<10x3x128x64xf32>, tensor<10x3x8x64xf32>) -> tensor<10x3x8x64xf32>
    return %1 : tensor<10x3x8x64xf32>
  }
}
