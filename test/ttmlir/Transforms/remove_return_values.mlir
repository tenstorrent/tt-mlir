// RUN: ttmlir-opt --ttir-remove-return-values -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @single_return_value
// CHECK-SAME: (%{{.*}}: tensor<64x128xf32>, %{{.*}}: tensor<64x128xf32>)
// CHECK-NOT: -> tensor<64x128xf32>
func.func @single_return_value(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = linalg.abs ins(%arg0 : tensor<64x128xf32>) outs(%arg1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: linalg.abs
  // CHECK: return
  // CHECK-NOT: return %
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @multiple_return_values
// CHECK-SAME: (%{{.*}}: tensor<64x128xf32>, %{{.*}}: tensor<64x128xf32>)
// CHECK-NOT: -> (tensor<64x128xf32>, tensor<64x128xf32>)
func.func @multiple_return_values(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> (tensor<64x128xf32>, tensor<64x128xf32>) {
  %0 = linalg.abs ins(%arg0 : tensor<64x128xf32>) outs(%arg1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %1 = linalg.exp ins(%arg0 : tensor<64x128xf32>) outs(%arg1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: linalg.abs
  // CHECK: linalg.exp
  // CHECK: return
  // CHECK-NOT: return %
  return %0, %1 : tensor<64x128xf32>, tensor<64x128xf32>
}

// CHECK-LABEL: func.func @already_void_return
// CHECK-SAME: (%{{.*}}: tensor<64x128xf32>, %{{.*}}: tensor<64x128xf32>)
// CHECK-NOT: ->
func.func @already_void_return(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) {
  %0 = linalg.abs ins(%arg0 : tensor<64x128xf32>) outs(%arg1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: linalg.abs
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @different_return_types
// CHECK-SAME: (%{{.*}}: tensor<64x128xf32>, %{{.*}}: i32)
// CHECK-NOT: -> (tensor<64x128xf32>, i32, f32)
func.func @different_return_types(%arg0: tensor<64x128xf32>, %arg1: i32) -> (tensor<64x128xf32>, i32, f32) {
  %cst = arith.constant 3.14 : f32
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.abs ins(%arg0 : tensor<64x128xf32>) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: arith.constant
  // CHECK: tensor.empty
  // CHECK: linalg.abs
  // CHECK: return
  // CHECK-NOT: return %
  return %1, %arg1, %cst : tensor<64x128xf32>, i32, f32
}
