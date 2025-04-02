// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module {
  func.func @reciprocal_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    // CHECK: %{{.*}} = linalg.reciprocal ins(%arg0 : tensor<64x128xf32>) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %1 = "ttir.reciprocal"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
