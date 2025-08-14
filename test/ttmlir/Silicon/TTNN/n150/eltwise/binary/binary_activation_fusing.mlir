// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-fusing-pass=true" -o %t %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t

module {
  func.func @simple_output_relu(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.relu"(%1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %3 : tensor<64x128xf32>
  }

  func.func @simple_lhs_relu(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.add"(%1, %arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %3 : tensor<64x128xf32>
  }

  func.func @simple_rhs_relu(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.relu"(%arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.add"(%arg0, %1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %3 : tensor<64x128xf32>
  }

  func.func @basic_activation_fusion(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.sigmoid"(%arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %4 = ttir.empty() : tensor<64x128xf32>
    %5 = "ttir.add"(%1, %3, %4) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %6 = ttir.empty() : tensor<64x128xf32>
    %7 = "ttir.tanh"(%5, %6) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %7 : tensor<64x128xf32>
  }

  func.func @parametric_and_multiple_activations(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.sigmoid"(%1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %4 = ttir.empty() : tensor<64x128xf32>
    %5 = "ttir.leaky_relu"(%arg1, %4) {parameter = 0.1 : f32} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %6 = ttir.empty() : tensor<64x128xf32>
    %7 = "ttir.multiply"(%3, %5, %6) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %8 = ttir.empty() : tensor<64x128xf32>
    %9 = "ttir.gelu"(%7, %8) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %10 = ttir.empty() : tensor<64x128xf32>
    %11 = "ttir.leaky_relu"(%9, %10) {parameter = 0.2 : f32} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %11 : tensor<64x128xf32>
  }
}
