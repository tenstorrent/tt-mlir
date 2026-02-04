// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// CHECK-LABEL: @equal_scalar_f32
func.func @equal_scalar_f32(%arg0: tensor<32x32xf32>) -> tensor<32x32xi1> {
  // CHECK: ttnn.equal_scalar
  // CHECK-NOT: ttir.equal_scalar
  %0 = "ttir.equal_scalar"(%arg0) <{scalar = 2.0 : f32}> : (tensor<32x32xf32>) -> tensor<32x32xi1>
  return %0 : tensor<32x32xi1>
}

// CHECK-LABEL: @equal_scalar_i32
func.func @equal_scalar_i32(%arg0: tensor<32x32xi32>) -> tensor<32x32xi1> {
  // CHECK: ttnn.equal_scalar
  // CHECK-NOT: ttir.equal_scalar
  %0 = "ttir.equal_scalar"(%arg0) <{scalar = 5 : i32}> : (tensor<32x32xi32>) -> tensor<32x32xi1>
  return %0 : tensor<32x32xi1>
}
