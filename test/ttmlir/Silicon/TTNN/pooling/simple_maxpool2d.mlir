// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// We mark max pool as channel last so we don't expect any permute ops.
module attributes {} {
  func.func @forward(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16> {
    %0 = tensor.empty() : tensor<1x64x64x32xbf16>
    // CHECK-NOT: "ttnn.permute"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.max_pool2d"
    // CHECK: "ttnn.reshape"
    %1 = "ttir.max_pool2d"(%arg0, %0) <{kernel_height=2: si32, kernel_width=2: si32, stride_height=2: si32, stride_width=2: si32, dilation_height=1: si32, dilation_width=1: si32, ceil_mode=false, padding_left=0: si32, padding_right=0: si32, padding_top=0: si32, padding_bottom=0: si32, channel_last = 1}> : (tensor<1x128x128x32xbf16>, tensor<1x64x64x32xbf16>) -> tensor<1x64x64x32xbf16>
    return %1 : tensor<1x64x64x32xbf16>
  }
}
