// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-fusing-pass=true enable-eltwise-activation-fusion=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // With fusion disabled, unary activations stay as standalone ops.
  // CHECK-LABEL: func.func @basic_activation_fusion_disabled
  func.func @basic_activation_fusion_disabled(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.relu"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    %1 = "ttir.sigmoid"(%arg1) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = "ttir.add"(%0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %3 = "ttir.tanh"(%2) : (tensor<64x128xf32>) -> tensor<64x128xf32>

    // CHECK: "ttnn.relu"
    // CHECK: "ttnn.sigmoid"
    // CHECK: "ttnn.add"
    // CHECK-SAME: activations = []
    // CHECK-SAME: input_tensor_a_activations = []
    // CHECK-SAME: input_tensor_b_activations = []
    // CHECK: "ttnn.tanh"

    return %3 : tensor<64x128xf32>
  }
}
