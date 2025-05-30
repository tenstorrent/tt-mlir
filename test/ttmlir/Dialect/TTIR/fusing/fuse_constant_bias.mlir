// RUN: ttmlir-opt --canonicalize %s | ttmlir-opt --ttir-fusing %s | FileCheck %s

// Fuse constant bias + reshape + add into conv. We also check that all uses of add are updated.
module {
  // CHECK-LABEL: func.func @conv2d_fuse
  func.func @conv2d_fuse(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<2x64x3x3xbf16>) -> tensor<1x30x30x2xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x2xbf16>
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"(%arg0, %arg1,
    // CHECK-SAME: dilation = 1
    // CHECK-SAME: groups = 1
    // CHECK-SAME: padding = 0
    // CHECK-SAME: stride = 1
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<2x64x3x3xbf16>, tensor<1x30x30x2xbf16>) -> tensor<1x30x30x2xbf16>
    // CHECK-NOT: "ttir.reshape"
    // CHECK-NOT: "ttir.add"
    %2 = ttir.empty() : tensor<1x30x30x2xbf16>
    %3 = "ttir.constant"() <{value = dense<[1.01, 2.02]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %4 = ttir.empty() : tensor<1x1x1x2xf32>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<2xf32>, tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32>
    %6 = "ttir.add"(%1, %5, %2) : (tensor<1x30x30x2xbf16>, tensor<1x1x1x2xf32>, tensor<1x30x30x2xbf16>) -> tensor<1x30x30x2xbf16>
    return %6: tensor<1x30x30x2xbf16>
  }
}
