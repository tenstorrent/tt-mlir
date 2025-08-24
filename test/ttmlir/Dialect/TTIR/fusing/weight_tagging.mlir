// RUN: ttmlir-opt --ttcore-register-device --ttir-fusing --mlir-print-local-scope --ttnn-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK: func.func @tagging
  // CHECK-SAME: tensor<64x64x3x3
  // CHECK-SAME: memref<12288x3xbf16, #ttnn.buffer_type<system_memory>
  func.func @tagging(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1: tensor<1x30x30x64xbf16>
  }
}
