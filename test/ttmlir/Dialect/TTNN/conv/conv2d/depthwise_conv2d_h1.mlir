// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// A depthwise conv2d over a unit-height input is the shape a framework
// produces when it decomposes a 1D convolution. Its config tensors must stay
// in L1, matching what Conv1dOpConversionPattern does for the native
// ttnn::conv1d path (tt-metal #45075, tt-mlir #9276).
module {
  func.func @depthwise_conv2d_unit_height(%arg0: tensor<1x1x32x64xbf16>, %arg1: tensor<64x1x1x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x30x64xbf16> {
    // CHECK-LABEL: func.func @depthwise_conv2d_unit_height
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: config_tensors_in_dram = false
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 64: i32
            }> : (tensor<1x1x32x64xbf16>, tensor<64x1x1x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x30x64xbf16>
    return %0 : tensor<1x1x30x64xbf16>
  }

  // A depthwise conv2d over a taller input keeps the general in-DRAM behavior,
  // so the workaround above does not regress it into OOM.
  func.func @depthwise_conv2d_tall(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x1x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK-LABEL: func.func @depthwise_conv2d_tall
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: config_tensors_in_dram = true
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 64: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x1x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %0 : tensor<1x30x30x64xbf16>
  }

  // A unit-height conv2d that is not depthwise also keeps the in-DRAM default:
  // the workaround is gated on the depthwise shape, not on the height alone.
  func.func @dense_conv2d_unit_height(%arg0: tensor<1x1x32x64xbf16>, %arg1: tensor<64x64x1x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x30x64xbf16> {
    // CHECK-LABEL: func.func @dense_conv2d_unit_height
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: config_tensors_in_dram = true
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x1x32x64xbf16>, tensor<64x64x1x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x30x64xbf16>
    return %0 : tensor<1x1x30x64xbf16>
  }
}
