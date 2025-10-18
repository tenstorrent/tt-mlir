// RUN: ttmlir-opt --convert-tosa-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_reshape(%arg0: tensor<63xf32>) -> tensor<1x3x3x7xf32> {
    %shape = "tosa.const_shape"() {values = dense<[1, 3, 3, 7]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.reshape %arg0, %shape : (tensor<63xf32>, !tosa.shape<4>) -> tensor<1x3x3x7xf32>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: shape = [1 : i32, 3 : i32, 3 : i32, 7 : i32]
    return %0 : tensor<1x3x3x7xf32>
  }

  func.func @test_reshape_2d_to_3d(%arg0: tensor<64x128xbf16>) -> tensor<2x32x128xbf16> {
    %shape = "tosa.const_shape"() {values = dense<[2, 32, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.reshape %arg0, %shape : (tensor<64x128xbf16>, !tosa.shape<3>) -> tensor<2x32x128xbf16>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: shape = [2 : i32, 32 : i32, 128 : i32]
    return %0 : tensor<2x32x128xbf16>
  }

  func.func @test_reshape_flatten(%arg0: tensor<4x8x16xf32>) -> tensor<512xf32> {
    %shape = "tosa.const_shape"() {values = dense<[512]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %shape : (tensor<4x8x16xf32>, !tosa.shape<1>) -> tensor<512xf32>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: shape = [512 : i32]
    return %0 : tensor<512xf32>
  }

  func.func @test_reshape_expand_dims(%arg0: tensor<32x32xi32>) -> tensor<1x32x32x1xi32> {
    %shape = "tosa.const_shape"() {values = dense<[1, 32, 32, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.reshape %arg0, %shape : (tensor<32x32xi32>, !tosa.shape<4>) -> tensor<1x32x32x1xi32>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: shape = [1 : i32, 32 : i32, 32 : i32, 1 : i32]
    return %0 : tensor<1x32x32x1xi32>
  }
}
