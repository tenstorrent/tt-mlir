// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-implicit-broadcast-folding-pass=false enable-repeat-folding-workaround-pass=false" %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"
    // CHECK-SAME: repeat_dims = #ttnn.shape<1x16x1>
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64 : 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %3 : tensor<1x16x32xf32>
  }
}

module {
  func.func public @main(%arg0: tensor<1xf32>, %arg1: tensor<512x512xf32>) -> (tensor<512x512xf32>) {
    // CHECK: %{{[0-9]+}} = "ttnn.reshape"
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"
    // CHECK-SAME: repeat_dims = #ttnn.shape<512x512>
    %0 = tensor.empty() : tensor<1x1xf32>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = tensor.empty() : tensor<512x512xf32>
    %3 = "ttir.broadcast"(%1, %2) <{broadcast_dimensions = array<i64 : 512, 512>}> : (tensor<1x1xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    %4 = tensor.empty() : tensor<512x512xf32>
    %5 = "ttir.maximum"(%3, %arg1, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    return %5 : tensor<512x512xf32>
  }
}

module {
    func.func @main(%arg0: tensor<1x23x40x1xf32>, %arg1: tensor<128xf32>) -> tensor<1x23x40x128xf32> {
      // CHECK: %{{[0-9]+}} = "ttnn.reshape"
      // CHECK: %{{[0-9]+}} = "ttnn.repeat"
      // CHECK-SAME: repeat_dims = #ttnn.shape<1x23x40x1>
      %0 = tensor.empty() : tensor<1x23x40x128xf32>
      %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64 : 1, 1, 1, 128>}> : (tensor<1x23x40x1xf32>, tensor<1x23x40x128xf32>) -> tensor<1x23x40x128xf32>
      %2 = tensor.empty() : tensor<1x1x1x128xf32>
      %3 = "ttir.reshape"(%arg1, %2) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
      %4 = tensor.empty() : tensor<1x23x40x128xf32>
      %5 = "ttir.broadcast"(%3, %4) <{broadcast_dimensions = array<i64 : 1, 23, 40, 1>}> : (tensor<1x1x1x128xf32>, tensor<1x23x40x128xf32>) -> tensor<1x23x40x128xf32>
      %6 = tensor.empty() : tensor<1x23x40x128xf32>
      %7 = "ttir.div"(%1, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x23x40x128xf32>, tensor<1x23x40x128xf32>, tensor<1x23x40x128xf32>) -> tensor<1x23x40x128xf32>
      return %7 : tensor<1x23x40x128xf32>
    }
  }

module {
    func.func @main(%arg0: tensor<6x2xf32>) -> tensor<2400x2xf32> {
      // CHECK: %{{[0-9]+}} = "ttnn.repeat"
      // CHECK-SAME: repeat_dims = #ttnn.shape<400x1x1x1>
      %0 = tensor.empty() : tensor<1x6x2xf32>
      %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 6 : i32, 2 : i32]}> : (tensor<6x2xf32>, tensor<1x6x2xf32>) -> tensor<1x6x2xf32>
      %2 = tensor.empty() : tensor<1x6x1x2xf32>
      %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 6 : i32, 1 : i32, 2 : i32]}> : (tensor<1x6x2xf32>, tensor<1x6x1x2xf32>) -> tensor<1x6x1x2xf32>
      %4 = tensor.empty() : tensor<400x6x1x2xf32>
      %5 = "ttir.broadcast"(%3, %4) <{broadcast_dimensions = array<i64: 400, 1, 1, 1>}> : (tensor<1x6x1x2xf32>, tensor<400x6x1x2xf32>) -> tensor<400x6x1x2xf32>
      %6 = tensor.empty() : tensor<2400x1x2xf32>
      %7 = "ttir.reshape"(%5, %6) <{shape = [2400 : i32, 1 : i32, 2 : i32]}> : (tensor<400x6x1x2xf32>, tensor<2400x1x2xf32>) -> tensor<2400x1x2xf32>
      %8 = tensor.empty() : tensor<2400x2xf32>
      %9 = "ttir.reshape"(%7, %8) <{shape = [2400 : i32, 2 : i32]}> : (tensor<2400x1x2xf32>, tensor<2400x2xf32>) -> tensor<2400x2xf32>
      return %9 : tensor<2400x2xf32>
    }
  }
