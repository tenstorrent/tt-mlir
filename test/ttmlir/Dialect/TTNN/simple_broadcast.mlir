// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s| FileCheck %s
module {
  func.func @main(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xi32>) -> tensor<1x16x32xf32> {
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.to_device"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"([[VAL0]])
    %0 = tensor.empty() : tensor<1x1x32xf32>
    %1 = "ttir.typecast"(%arg1, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1x32xi32>, tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.broadcast"(%arg0, %2) <{dimension = [1, 2]}> : (tensor<1x16x1xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %4 = tensor.empty() : tensor<1x16x32xf32>
    %5 = "ttir.broadcast"(%1, %4) <{dimension = [0, 1, 2]}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %6 = tensor.empty() : tensor<1x16x32xf32>
    %7 = "ttir.multiply"(%3, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %7 : tensor<1x16x32xf32>
  }
}

module {
  func.func @main(%arg0: tensor<1x10xi32>, %arg1: tensor<10x1xi32>) -> tensor<10x10xi32> {
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.to_device"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"([[VAL0]])
    %0 = tensor.empty() : tensor<10x10xi32>
    %1 = "ttir.broadcast"(%arg0, %0) <{dimension = [0, 1]}> : (tensor<1x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
    %2 = tensor.empty() : tensor<10x10xi32>
    %3 = "ttir.broadcast"(%arg1, %2) <{dimension = [0, 1]}> : (tensor<10x1xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
    %4 = tensor.empty() : tensor<10x10xi32>
    %5 = "ttir.subtract"(%1, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<10x10xi32>, tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
    return %5 : tensor<10x10xi32>
  }
}
