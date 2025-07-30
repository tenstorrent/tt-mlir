// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true
// Currently unsupported as test fails in runtime when ceil_mode is true and dilation is specified.
// tt-Metal issue: https://github.com/tenstorrent/tt-metal/issues/25894
module {
  // Test 5: MaxPool2dOp with flattened compat info and with kernel, stride, dilation, padding and ceil_mode specified
  func.func @max_pool2d_flattened_with_kernel_stride_dilation_padding_ceil(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16> {
    %0 = ttir.empty() : tensor<1x17x17x64xbf16>
    // CHECK: ttnn.max_pool2d
    %1 = "ttir.max_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 2, 2>,
      padding = array<i32: 2, 2, 2, 2>,
      ceil_mode = true
    }> : (tensor<1x32x32x64xbf16>, tensor<1x17x17x64xbf16>) -> tensor<1x17x17x64xbf16>
    return %1 : tensor<1x17x17x64xbf16>
  }
}
