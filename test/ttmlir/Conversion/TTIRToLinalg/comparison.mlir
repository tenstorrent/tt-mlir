// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module attributes {} {
  // Test for equal operation with integer type
  func.func @equal_int_test(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi1> {
    // CHECK: tensor.empty() : tensor<64x128xi1>
    %0 = ttir.empty() : tensor<64x128xi1>

    // CHECK: linalg.generic {{.*}} ins(%arg0, %arg1 : tensor<64x128xi32>, tensor<64x128xi32>) outs({{.*}} : tensor<64x128xi1>) {
    // CHECK: ^bb0({{.*}}: i32, {{.*}}: i32, {{.*}}: i1):
    // CHECK:   arith.cmpi eq, {{.*}}, {{.*}} : i32
    // CHECK:   linalg.yield {{.*}} : i1
    // CHECK: } -> tensor<64x128xi1>

    %1 = "ttir.eq"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi1>) -> tensor<64x128xi1>

    return %1 : tensor<64x128xi1>
  }

  // Test for equal operation with float type
  func.func @equal_float_test(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xi1> {
    // CHECK: tensor.empty() : tensor<64x128xi1>
    %0 = ttir.empty() : tensor<64x128xi1>

    // CHECK: linalg.generic {{.*}} ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>) outs({{.*}} : tensor<64x128xi1>) {
    // CHECK: ^bb0({{.*}}: f32, {{.*}}: f32, {{.*}}: i1):
    // CHECK:   arith.cmpf oeq, {{.*}}, {{.*}} : f32
    // CHECK:   linalg.yield {{.*}} : i1
    // CHECK: } -> tensor<64x128xi1>

    %1 = "ttir.eq"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xi1>) -> tensor<64x128xi1>

    return %1 : tensor<64x128xi1>
  }

  // Test for greater than operation with integer type
  func.func @greater_than_int_test(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi1> {
    // CHECK: tensor.empty() : tensor<64x128xi1>
    %0 = ttir.empty() : tensor<64x128xi1>

    // CHECK: linalg.generic {{.*}} ins(%arg0, %arg1 : tensor<64x128xi32>, tensor<64x128xi32>) outs({{.*}} : tensor<64x128xi1>) {
    // CHECK: ^bb0({{.*}}: i32, {{.*}}: i32, {{.*}}: i1):
    // CHECK:   arith.cmpi sgt, {{.*}}, {{.*}} : i32
    // CHECK:   linalg.yield {{.*}} : i1
    // CHECK: } -> tensor<64x128xi1>

    %1 = "ttir.gt"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi1>) -> tensor<64x128xi1>

    return %1 : tensor<64x128xi1>
  }

  // Test for greater equal operation with float type
  func.func @greater_equal_float_test(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xi1> {
    // CHECK: tensor.empty() : tensor<64x128xi1>
    %0 = ttir.empty() : tensor<64x128xi1>

    // CHECK: linalg.generic {{.*}} ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>) outs({{.*}} : tensor<64x128xi1>) {
    // CHECK: ^bb0({{.*}}: f32, {{.*}}: f32, {{.*}}: i1):
    // CHECK:   arith.cmpf oge, {{.*}}, {{.*}} : f32
    // CHECK:   linalg.yield {{.*}} : i1
    // CHECK: } -> tensor<64x128xi1>

    %1 = "ttir.ge"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xi1>) -> tensor<64x128xi1>

    return %1 : tensor<64x128xi1>
  }
}
