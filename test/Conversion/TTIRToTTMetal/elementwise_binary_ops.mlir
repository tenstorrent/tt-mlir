// RUN: ttmlir-opt --ttir-to-ttmetal-elementwise-binary %s | FileCheck %s

// Test basic elementwise binary operations conversion

// CHECK-LABEL: @test_add
func.func @test_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "add"
  %0 = ttir.empty() : tensor<2x3xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: @test_subtract
func.func @test_subtract(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "subtract"
  %0 = ttir.empty() : tensor<4x4xf32>
  %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// CHECK-LABEL: @test_multiply
func.func @test_multiply(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "multiply"
  %0 = ttir.empty() : tensor<3x3xf32>
  %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// CHECK-LABEL: @test_divide
func.func @test_divide(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "divide"
  %0 = ttir.empty() : tensor<2x2xf32>
  %1 = "ttir.divide"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// Test comparison operations
// CHECK-LABEL: @test_greater_than
func.func @test_greater_than(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "greater_than"
  %0 = ttir.empty() : tensor<2x2xi1>
  %1 = "ttir.greater_than"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xi1>) -> tensor<2x2xi1>
  return %1 : tensor<2x2xi1>
}

// CHECK-LABEL: @test_less_than
func.func @test_less_than(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "less_than"
  %0 = ttir.empty() : tensor<2x2xi1>
  %1 = "ttir.less_than"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xi1>) -> tensor<2x2xi1>
  return %1 : tensor<2x2xi1>
}

// CHECK-LABEL: @test_equal
func.func @test_equal(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "equal"
  %0 = ttir.empty() : tensor<2x2xi1>
  %1 = "ttir.equal"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xi1>) -> tensor<2x2xi1>
  return %1 : tensor<2x2xi1>
}

// Test max/min operations
// CHECK-LABEL: @test_maximum
func.func @test_maximum(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "maximum"
  %0 = ttir.empty() : tensor<3x3xf32>
  %1 = "ttir.maximum"(%arg0, %arg1, %0) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// CHECK-LABEL: @test_minimum
func.func @test_minimum(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "minimum"
  %0 = ttir.empty() : tensor<3x3xf32>
  %1 = "ttir.minimum"(%arg0, %arg1, %0) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// Test with different element types
// CHECK-LABEL: @test_add_f64
func.func @test_add_f64(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "add"
  %0 = ttir.empty() : tensor<2x2xf64>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x2xf64>, tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  return %1 : tensor<2x2xf64>
}

// CHECK-LABEL: @test_add_i32
func.func @test_add_i32(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: ttmetal.elementwise_binary
  // CHECK-SAME: op_type = "add"
  %0 = ttir.empty() : tensor<2x2xi32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %1 : tensor<2x2xi32>
}

// Test broadcasting
// CHECK-LABEL: @test_broadcast
func.func @test_broadcast(%arg0: tensor<1x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: ttmetal.elementwise_binary
  %0 = ttir.empty() : tensor<3x4xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}
