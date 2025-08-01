// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_reshape(%arg0: tensor<63xf32>) -> tensor<1x3x3x7xf32> {
    // CHECK: func.func {{.+}} [[IN_SIZE:tensor<[0-9]+xf32>]]
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 3, 3, 7>} : (tensor<63xf32>) -> tensor<1x3x3x7xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = tensor.empty() : [[OUT_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xf32>]]
    // CHECK: %[[VAL:[0-9]+]] = "ttir.reshape"(%arg{{[0-9]+}}, %[[OP_OUT]]){{.+}} ([[IN_SIZE]], [[OUT_SIZE]]) -> [[OUT_SIZE]]
    // CHECK: return %[[VAL]] : [[OUT_SIZE]]
    return %0 : tensor<1x3x3x7xf32>
  }

  func.func @test_reshape_2d_to_3d(%arg0: tensor<64x128xbf16>) -> tensor<2x32x128xbf16> {
    // CHECK: func.func @test_reshape_2d_to_3d(%arg0: [[IN_2D:tensor<64x128xbf16>]]) -> [[OUT_3D:tensor<2x32x128xbf16>]]
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 2, 32, 128>} : (tensor<64x128xbf16>) -> tensor<2x32x128xbf16>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : [[OUT_3D]]
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %[[EMPTY]]) <{shape = [2 : i32, 32 : i32, 128 : i32], operand_constraints = [#{{.*}}, #{{.*}}]}> : ([[IN_2D]], [[OUT_3D]]) -> [[OUT_3D]]
    return %0 : tensor<2x32x128xbf16>
  }

  func.func @test_reshape_flatten(%arg0: tensor<4x8x16xf32>) -> tensor<512xf32> {
    // CHECK: func.func @test_reshape_flatten(%arg0: [[IN_3D:tensor<4x8x16xf32>]]) -> [[OUT_1D:tensor<512xf32>]]
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 512>} : (tensor<4x8x16xf32>) -> tensor<512xf32>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : [[OUT_1D]]
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %[[EMPTY]]) <{shape = [512 : i32], operand_constraints = [#{{.*}}, #{{.*}}]}> : ([[IN_3D]], [[OUT_1D]]) -> [[OUT_1D]]
    return %0 : tensor<512xf32>
  }

  func.func @test_reshape_expand_dims(%arg0: tensor<32x32xi32>) -> tensor<1x32x32x1xi32> {
    // CHECK: func.func @test_reshape_expand_dims(%arg0: [[IN_2D:tensor<32x32xi32>]]) -> [[OUT_4D:tensor<1x32x32x1xi32>]]
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 32, 32, 1>} : (tensor<32x32xi32>) -> tensor<1x32x32x1xi32>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : [[OUT_4D]]
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %[[EMPTY]]) <{shape = [1 : i32, 32 : i32, 32 : i32, 1 : i32], operand_constraints = [#{{.*}}, #{{.*}}]}> : ([[IN_2D]], [[OUT_4D]]) -> [[OUT_4D]]
    return %0 : tensor<1x32x32x1xi32>
  }
}
