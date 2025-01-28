// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-repeat-folding-workaround-pass=false" %s | FileCheck %s

module {
  func.func @repeat_on_one_dim(%arg0: tensor<1x32x32xf32>) -> tensor<32x32x32xf32> {
    // CHECK: "ttnn.repeat"
    // CHECK-SAME: repeat_dims = #ttnn.shape<32x1x1>
    %0 = tensor.empty() : tensor<32x32x32xf32>
    %1 = "ttir.repeat"(%arg0, %0) {repeat_dimensions = array<i64: 32, 1, 1>} : (tensor<1x32x32xf32>, tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
    return %1 : tensor<32x32x32xf32>
  }

  func.func @repeat_on_all_dims(%arg0: tensor<1x1x32xf32>) -> tensor<32x32x64xf32> {
    // CHECK: "ttnn.repeat"
    // CHECK-SAME: repeat_dims = #ttnn.shape<32x32x2>
    %0 = tensor.empty() : tensor<32x32x64xf32>
    %1 = "ttir.repeat"(%arg0, %0) {repeat_dimensions = array<i64: 32, 32, 2>} : (tensor<1x1x32xf32>, tensor<32x32x64xf32>) -> tensor<32x32x64xf32>
    return %1 : tensor<32x32x64xf32>
  }
}
