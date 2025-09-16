// RUN: ttmlir-opt --ttcore-register-device --ttir-flatten-sliding-window --ttnn-layout --convert-ttir-to-ttnn --ttnn-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Test fusion of Conv2d with ReLU activation.
  // CHECK-LABEL: func.func @conv2d_with_relu
  func.func @conv2d_with_relu(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[CONV:.*]] = "ttnn.conv2d"
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // CHECK-NOT: ttnn.relu
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %3 = "ttir.relu"(%1, %2) : (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // This reshape is comming from flattening sliding window.
    // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[CONV]])

    // CHECK: return %[[RESHAPE]]
    return %3 : tensor<1x30x30x64xbf16>
  }

  // Test fusion of Conv2d with ReLU6 activation.
  // CHECK-LABEL: func.func @conv2d_with_relu6
  func.func @conv2d_with_relu6(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[CONV:.*]] = "ttnn.conv2d"
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // CHECK-NOT: ttnn.relu6
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %3 = "ttir.relu6"(%1, %2) : (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // This reshape is comming from flattening sliding window.
    // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[CONV]])

    // CHECK: return %[[RESHAPE]]
    return %3 : tensor<1x30x30x64xbf16>
  }

  // Test that we cannot fuse Conv2d and ReLU because Conv2d has multiple uses.
  // CHECK-LABEL: func.func @conv2d_with_multiple_uses
  func.func @conv2d_with_multiple_uses(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %{{.*}} = "ttnn.conv2d"
    // CHECK-NOT: activation = <op_type = relu
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // CHECK: %[[RELU:.*]] = "ttnn.relu"
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %3 = "ttir.relu"(%1, %2) : (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // CHECK: %[[ADD:.*]] = "ttnn.add"
    %4 = ttir.empty() : tensor<1x30x30x64xbf16>
    // Second use of conv2d, we cannot fuse.
    %5 = "ttir.add"(%1, %3, %4) : (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    return %5 :tensor<1x30x30x64xbf16>
  }

  // Test that we cannot fuse Conv2d and Sigmoid because Conv2d only allows ReLU activation.
  // CHECK-LABEL: func.func @conv2d_with_sigmoid
  func.func @conv2d_with_sigmoid(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %{{.*}} = "ttnn.conv2d"
    // CHECK-NOT: activation = <op_type = sigmoid
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // CHECK: %[[SIGMOID:.*]] = "ttnn.sigmoid"
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>

    // Sigmoid cannot be fused with conv2d.
    %3 = "ttir.sigmoid"(%1, %2) : (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // CHECK: return %[[SIGMOID]]
    return %3 : tensor<1x30x30x64xbf16>
  }
}
