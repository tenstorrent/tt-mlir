// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module {
  func.func @main(%arg0: tensor<1x23x40x1xf32>, %arg1: tensor<128xf32>) -> tensor<1x23x40x128xf32> {
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.to_device"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"([[VAL0]])
    %0 = tensor.empty() : tensor<1x23x40x128xf32>
    %1 = "ttir.broadcast"(%arg0, %0) <{dimension = [0, 1, 2, 3]}> : (tensor<1x23x40x1xf32>, tensor<1x23x40x128xf32>) -> tensor<1x23x40x128xf32>
    %2 = tensor.empty() : tensor<1x23x40x128xf32>
    %3 = "ttir.broadcast"(%arg1, %2) <{dimension = [3]}> : (tensor<128xf32>, tensor<1x23x40x128xf32>) -> tensor<1x23x40x128xf32>
    %4 = tensor.empty() : tensor<1x23x40x128xf32>
    %5 = "ttir.div"(%1, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x23x40x128xf32>, tensor<1x23x40x128xf32>, tensor<1x23x40x128xf32>) -> tensor<1x23x40x128xf32>
    return %5 : tensor<1x23x40x128xf32>
  }
}

module {
  func.func @main(%arg0: tensor<32xi32>, %arg1: tensor<32x1xi32>) -> tensor<32x32xbf16> {
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.to_device"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"([[VAL0]])
    %0 = tensor.empty() : tensor<32x32xi32>
    %1 = "ttir.broadcast"(%arg0, %0) <{dimension = [1]}> : (tensor<32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %2 = tensor.empty() : tensor<32x32xi32>
    %3 = "ttir.broadcast"(%arg1, %2) <{dimension = [0, 1]}> : (tensor<32x1xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %4 = tensor.empty() : tensor<32x32xbf16>
    %5 = "ttir.gt"(%1, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %5 : tensor<32x32xbf16>
  }
}

module {
  func.func @main(%arg0: tensor<16x1xf32>, %arg1: tensor<1x1x32xi32>) -> tensor<1x16x32xf32> {
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.to_device"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"([[VAL0]])
    %0 = tensor.empty() : tensor<1x1x32xf32>
    %1 = "ttir.typecast"(%arg1, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x1x32xi32>, tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.broadcast"(%arg0, %2) <{dimension = [1, 2]}> : (tensor<16x1xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
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

module {
  func.func @main(%arg0: tensor<6x2xf32>) -> tensor<2400x2xf32> {
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.to_device"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"([[VAL0]])
    %0 = tensor.empty() : tensor<1x6x2xf32>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 6 : i32, 2 : i32]}> : (tensor<6x2xf32>, tensor<1x6x2xf32>) -> tensor<1x6x2xf32>
    %2 = tensor.empty() : tensor<1x6x1x2xf32>
    %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 6 : i32, 1 : i32, 2 : i32]}> : (tensor<1x6x2xf32>, tensor<1x6x1x2xf32>) -> tensor<1x6x1x2xf32>
    %4 = tensor.empty() : tensor<400x6x1x2xf32>
    %5 = "ttir.broadcast"(%3, %4) <{dimension = [0, 1, 2, 3]}> : (tensor<1x6x1x2xf32>, tensor<400x6x1x2xf32>) -> tensor<400x6x1x2xf32>
    %6 = tensor.empty() : tensor<2400x1x2xf32>
    %7 = "ttir.reshape"(%5, %6) <{shape = [2400 : i32, 1 : i32, 2 : i32]}> : (tensor<400x6x1x2xf32>, tensor<2400x1x2xf32>) -> tensor<2400x1x2xf32>
    %8 = tensor.empty() : tensor<2400x2xf32>
    %9 = "ttir.reshape"(%7, %8) <{shape = [2400 : i32, 2 : i32]}> : (tensor<2400x1x2xf32>, tensor<2400x2xf32>) -> tensor<2400x2xf32>
    return %9 : tensor<2400x2xf32>
  }
}
