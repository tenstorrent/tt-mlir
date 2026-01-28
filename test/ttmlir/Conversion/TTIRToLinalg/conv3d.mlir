// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test 1: Conv3dOp basic with 3x3x3 kernel
module {
  func.func @conv3d_basic(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<16x4x3x3x3xbf16>, %arg2: tensor<1x1x1x1x16xbf16>) -> tensor<1x6x26x26x16xbf16> {
    // CHECK: tosa.transpose
    // CHECK: linalg.fill
    // CHECK: linalg.conv_3d_ndhwc_dhwcf
    // CHECK: tosa.add
    // CHECK-NOT: ttir.conv3d
    %1 = "ttir.conv3d"(%arg0, %arg1, %arg2) <{
      stride = array<i32: 1, 1, 1>,
      padding = array<i32: 0, 0, 0>,
      groups = 1 : i32,
      padding_mode = "zeros"
    }> : (tensor<1x8x28x28x4xbf16>, tensor<16x4x3x3x3xbf16>, tensor<1x1x1x1x16xbf16>) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}

// Test 2: Conv3dOp with stride 2
module {
  func.func @conv3d_stride2(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<16x4x3x3x3xbf16>, %arg2: tensor<1x1x1x1x16xbf16>) -> tensor<1x3x13x13x16xbf16> {
    // CHECK: tosa.transpose
    // CHECK: linalg.fill
    // CHECK: linalg.conv_3d_ndhwc_dhwcf
    // CHECK: tosa.add
    %1 = "ttir.conv3d"(%arg0, %arg1, %arg2) <{
      stride = array<i32: 2, 2, 2>,
      padding = array<i32: 0, 0, 0>,
      groups = 1 : i32,
      padding_mode = "zeros"
    }> : (tensor<1x8x28x28x4xbf16>, tensor<16x4x3x3x3xbf16>, tensor<1x1x1x1x16xbf16>) -> tensor<1x3x13x13x16xbf16>
    return %1 : tensor<1x3x13x13x16xbf16>
  }
}

// Test 3: Conv3dOp without bias
module {
  func.func @conv3d_no_bias(%arg0: tensor<1x8x28x28x4xbf16>, %arg1: tensor<16x4x3x3x3xbf16>) -> tensor<1x6x26x26x16xbf16> {
    // CHECK: tosa.transpose
    // CHECK: linalg.fill
    // CHECK: linalg.conv_3d_ndhwc_dhwcf
    // CHECK-NOT: tosa.add
    %1 = "ttir.conv3d"(%arg0, %arg1) <{
      stride = array<i32: 1, 1, 1>,
      padding = array<i32: 0, 0, 0>,
      groups = 1 : i32,
      padding_mode = "zeros"
    }> : (tensor<1x8x28x28x4xbf16>, tensor<16x4x3x3x3xbf16>) -> tensor<1x6x26x26x16xbf16>
    return %1 : tensor<1x6x26x26x16xbf16>
  }
}
