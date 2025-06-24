// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-trace=true" -o mnist_linear_out.mlir %s
// RUN: FileCheck %s --input-file=mnist_linear_out.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer mnist_linear_out.mlir > %t.ttnn
module @MNISTLinear attributes {} {
  func.func @forward(%arg0: tensor<1x784xbf16>, %arg1: tensor<784x256xbf16>, %arg2: tensor<256xbf16>, %arg3: tensor<256x10xbf16>, %arg4: tensor<10xbf16>) -> tensor<1x10xbf16> {
    // CHECK: ttnn.trace
    %0 = ttir.empty() : tensor<1x256xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x784xbf16>, tensor<784x256xbf16>, tensor<1x256xbf16>) -> tensor<1x256xbf16>
    %2 = ttir.empty() : tensor<1x256xbf16>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<1x256xbf16>, tensor<256xbf16>, tensor<1x256xbf16>) -> tensor<1x256xbf16>
    %4 = ttir.empty() : tensor<1x256xbf16>
    %5 = "ttir.relu"(%3, %4) : (tensor<1x256xbf16>, tensor<1x256xbf16>) -> tensor<1x256xbf16>
    %6 = ttir.empty() : tensor<1x10xbf16>
    %7 = "ttir.matmul"(%5, %arg3, %6) : (tensor<1x256xbf16>, tensor<256x10xbf16>, tensor<1x10xbf16>) -> tensor<1x10xbf16>
    %8 = ttir.empty() : tensor<1x10xbf16>
    %9 = "ttir.add"(%7, %arg4, %8) : (tensor<1x10xbf16>, tensor<10xbf16>, tensor<1x10xbf16>) -> tensor<1x10xbf16>
    return %9 : tensor<1x10xbf16>
  }
}
