// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// CHECK-LABEL: @pow_scalar_f32
func.func @pow_scalar_f32(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: ttnn.pow_scalar
  // CHECK-NOT: ttir.pow_scalar
  %0 = "ttir.pow_scalar"(%arg0) <{scalar = 2.0 : f32}> : (tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: @pow_scalar_i32
func.func @pow_scalar_i32(%arg0: tensor<32x32xi32>) -> tensor<32x32xi32> {
  // CHECK: ttnn.pow_scalar
  // CHECK-NOT: ttir.pow_scalar
  %0 = "ttir.pow_scalar"(%arg0) <{scalar = 3 : i32}> : (tensor<32x32xi32>) -> tensor<32x32xi32>
  return %0 : tensor<32x32xi32>
}
