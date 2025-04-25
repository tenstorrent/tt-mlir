// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module attributes {} {
  func.func @logical_not_test(%arg0: tensor<64x128xi1>) -> tensor<64x128xi1> {
    // CHECK: tensor.empty() : tensor<64x128xi1>
    %0 = ttir.empty() : tensor<64x128xi1>

    // CHECK: linalg.generic {{.*}} ins(%arg0 : tensor<64x128xi1>) outs({{.*}} : tensor<64x128xi1>) {
    // CHECK: ^bb0({{.*}}: i1, {{.*}}: i1):
    // CHECK:   arith.constant true
    // CHECK:   arith.xori {{.*}}, {{.*}} : i1
    // CHECK:   linalg.yield {{.*}} : i1
    // CHECK: } -> tensor<64x128xi1>

    %1 = "ttir.logical_not"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xi1>, tensor<64x128xi1>) -> tensor<64x128xi1>

    return %1 : tensor<64x128xi1>
  }
}
