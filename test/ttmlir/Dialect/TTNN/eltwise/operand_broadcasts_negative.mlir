// RUN: not ttmlir-opt --ttir-load-system-desc --ttir-layout --convert-ttir-to-ttnn %s 2>&1 | FileCheck %s
// CHECK: error: 'ttir.multiply' op Operands are not broadcast compatible
module attributes {} {
  func.func @bcast_one_dim(%arg0: tensor<2x64x128xf32>, %arg1: tensor<4x64x128xf32>) -> tensor<4x64x128xf32> {
    %0 = tensor.empty() : tensor<4x64x128xf32>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2x64x128xf32>, tensor<4x64x128xf32>, tensor<4x64x128xf32>) -> tensor<4x64x128xf32>
    return %1 : tensor<4x64x128xf32>
  }
}
