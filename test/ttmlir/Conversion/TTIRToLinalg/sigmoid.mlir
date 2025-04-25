// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module attributes {} {
  func.func @sigmoid_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: tensor.empty() : tensor<64x128xf32>
    %0 = ttir.empty() : tensor<64x128xf32>

    // CHECK: linalg.generic {{.*}} ins(%arg0 : tensor<64x128xf32>) outs({{.*}} : tensor<64x128xf32>) {
    // CHECK: ^bb0({{.*}}: f32, {{.*}}: f32):
    // CHECK:   arith.negf {{.*}} : f32
    // CHECK:   math.exp {{.*}} : f32
    // CHECK:   arith.constant 1.000000e+00 : f32
    // CHECK:   arith.addf {{.*}}, {{.*}} : f32
    // CHECK:   arith.divf {{.*}}, {{.*}} : f32
    // CHECK:   linalg.yield {{.*}} : f32
    // CHECK: } -> tensor<64x128xf32>

    %1 = "ttir.sigmoid"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    return %1 : tensor<64x128xf32>
  }
}
