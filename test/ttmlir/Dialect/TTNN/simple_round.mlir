// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s| FileCheck %s
module attributes {} {
  func.func @roundnearesteven(%arg0: tensor<4xbf16>) -> tensor<4xbf16> {
    %0 = tensor.empty() : tensor<4xbf16>
    // CHECK: %[[C:.*]] = "ttnn.round"[[C:.*]]
    %1 = "ttir.roundnearesteven"(%arg0, %0) <{decimals = 0 : i32}> : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xbf16>
    return %1 : tensor<4xbf16>
  }
  func.func @round(%arg0: tensor<4xbf16>) -> tensor<4xbf16> {
    %0 = tensor.empty() : tensor<4xbf16>
    // CHECK: %[[C:.*]] = "ttnn.round"[[C:.*]]
    %1 = "ttir.round"(%arg0, %0) <{decimals = 1 : i32}> : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xbf16>
    return %1 : tensor<4xbf16>
  }
}
