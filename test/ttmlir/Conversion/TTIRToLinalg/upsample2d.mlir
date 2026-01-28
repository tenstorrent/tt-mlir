// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test 1: Upsample2dOp with nearest neighbor mode and scale factor 2
module {
  func.func @upsample2d_nearest_2x(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x64x64x64xbf16> {
    // CHECK: tosa.resize
    // CHECK-NOT: ttir.upsample2d
    %1 = "ttir.upsample2d"(%arg0) <{
      scale_factor = array<i32: 2, 2>,
      mode = "nearest"
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x64x64x64xbf16>
    return %1 : tensor<1x64x64x64xbf16>
  }
}

// Test 2: Upsample2dOp with bilinear mode and scale factor 2
module {
  func.func @upsample2d_bilinear_2x(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x64x64x64xbf16> {
    // CHECK: tosa.resize
    %1 = "ttir.upsample2d"(%arg0) <{
      scale_factor = array<i32: 2, 2>,
      mode = "bilinear"
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x64x64x64xbf16>
    return %1 : tensor<1x64x64x64xbf16>
  }
}

// Test 3: Upsample2dOp with different scale factors for height and width
module {
  func.func @upsample2d_asymmetric(%arg0: tensor<1x16x32x64xbf16>) -> tensor<1x64x64x64xbf16> {
    // CHECK: tosa.resize
    %1 = "ttir.upsample2d"(%arg0) <{
      scale_factor = array<i32: 4, 2>,
      mode = "nearest"
    }> : (tensor<1x16x32x64xbf16>) -> tensor<1x64x64x64xbf16>
    return %1 : tensor<1x64x64x64xbf16>
  }
}

// Test 4: Upsample2dOp with single scale factor (as array)
module {
  func.func @upsample2d_single_scale(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x96x96x64xbf16> {
    // CHECK: tosa.resize
    %1 = "ttir.upsample2d"(%arg0) <{
      scale_factor = array<i32: 3, 3>,
      mode = "nearest"
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x96x96x64xbf16>
    return %1 : tensor<1x96x96x64xbf16>
  }
}
