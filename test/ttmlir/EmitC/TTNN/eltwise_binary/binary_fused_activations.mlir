
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

// Test binary operations with post_activation
func.func @add_post_activation(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.relu"(%1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %3 : tensor<64x128xf32>
}

func.func @multiply_post_activation(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.sigmoid"(%1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %3 : tensor<64x128xf32>
}

func.func @subtract_post_activation(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.tanh"(%1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %3 : tensor<64x128xf32>
}

// Test binary operations with lhs activations
func.func @add_lhs_activation(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.add"(%1, %arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %3 : tensor<64x128xf32>
}

func.func @multiply_lhs_activation(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.gelu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.multiply"(%1, %arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %3 : tensor<64x128xf32>
}

// Test binary operations with rhs activations
func.func @add_rhs_activation(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.sigmoid"(%arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.add"(%arg0, %1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %3 : tensor<64x128xf32>
}

func.func @div_rhs_activation(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.leaky_relu"(%arg1, %0) {parameter = 0.1 : f32} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.div"(%arg0, %1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %3 : tensor<64x128xf32>
}

// Test binary operations with both lhs and rhs activations
func.func @add_both_activations(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.sigmoid"(%arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %4 = ttir.empty() : tensor<64x128xf32>
  %5 = "ttir.add"(%1, %3, %4) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %5 : tensor<64x128xf32>
}

func.func @maximum_both_activations(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.tanh"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.gelu"(%arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %4 = ttir.empty() : tensor<64x128xf32>
  %5 = "ttir.maximum"(%1, %3, %4) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %5 : tensor<64x128xf32>
}

// Test combined pre and post activations
func.func @multiply_pre_post_activations(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.sigmoid"(%arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %4 = ttir.empty() : tensor<64x128xf32>
  %5 = "ttir.multiply"(%1, %3, %4) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %6 = ttir.empty() : tensor<64x128xf32>
  %7 = "ttir.tanh"(%5, %6) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %7 : tensor<64x128xf32>
}

// Test binary composite operations with post_activation
func.func @matmul_post_activation(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
  %0 = ttir.empty() : tensor<64x96xbf16>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
  %2 = ttir.empty() : tensor<64x96xbf16>
  %3 = "ttir.relu"(%1, %2) : (tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
  return %3 : tensor<64x96xbf16>
}

func.func @linear_post_activation(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<32x128xf32>) -> tensor<32x128xf32> {
  %0 = ttir.empty() : tensor<32x128xf32>
  %1 = "ttir.linear"(%arg0, %arg1, %arg2, %0) : (tensor<32x64xf32>, tensor<64x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
  %2 = ttir.empty() : tensor<32x128xf32>
  %3 = "ttir.gelu"(%1, %2) : (tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
  return %3 : tensor<32x128xf32>
}

// Test binary composite operations with lhs activations
func.func @matmul_lhs_activation(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
  %0 = ttir.empty() : tensor<64x128xbf16>
  %1 = "ttir.sigmoid"(%arg0, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  %2 = ttir.empty() : tensor<64x96xbf16>
  %3 = "ttir.matmul"(%1, %arg1, %2) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
  return %3 : tensor<64x96xbf16>
}

// Test binary composite operations with rhs activations
func.func @matmul_rhs_activation(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
  %0 = ttir.empty() : tensor<128x96xbf16>
  %1 = "ttir.tanh"(%arg1, %0) : (tensor<128x96xbf16>, tensor<128x96xbf16>) -> tensor<128x96xbf16>
  %2 = ttir.empty() : tensor<64x96xbf16>
  %3 = "ttir.matmul"(%arg0, %1, %2) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
  return %3 : tensor<64x96xbf16>
}

// Test binary composite operations with both lhs and rhs activations
func.func @matmul_both_activations(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
  %0 = ttir.empty() : tensor<64x128xbf16>
  %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  %2 = ttir.empty() : tensor<128x96xbf16>
  %3 = "ttir.sigmoid"(%arg1, %2) : (tensor<128x96xbf16>, tensor<128x96xbf16>) -> tensor<128x96xbf16>
  %4 = ttir.empty() : tensor<64x96xbf16>
  %5 = "ttir.matmul"(%1, %3, %4) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
  return %5 : tensor<64x96xbf16>
}

// Test binary composite operations with pre and post activations
func.func @matmul_pre_post_activations(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
  %0 = ttir.empty() : tensor<64x128xbf16>
  %1 = "ttir.gelu"(%arg0, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  %2 = ttir.empty() : tensor<128x96xbf16>
  %3 = "ttir.leaky_relu"(%arg1, %2) {parameter = 0.2 : f32} : (tensor<128x96xbf16>, tensor<128x96xbf16>) -> tensor<128x96xbf16>
  %4 = ttir.empty() : tensor<64x96xbf16>
  %5 = "ttir.matmul"(%1, %3, %4) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
  %6 = ttir.empty() : tensor<64x96xbf16>
  %7 = "ttir.sigmoid"(%5, %6) : (tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
  return %7 : tensor<64x96xbf16>
}

// Test multiple activation chains
func.func @add_multiple_activations(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  %3 = "ttir.sigmoid"(%1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %4 = ttir.empty() : tensor<64x128xf32>
  %5 = "ttir.tanh"(%arg1, %4) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %6 = ttir.empty() : tensor<64x128xf32>
  %7 = "ttir.add"(%3, %5, %6) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %8 = ttir.empty() : tensor<64x128xf32>
  %9 = "ttir.gelu"(%7, %8) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %10 = ttir.empty() : tensor<64x128xf32>
  %11 = "ttir.leaky_relu"(%9, %10) {parameter = 0.1 : f32} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %11 : tensor<64x128xf32>
}
