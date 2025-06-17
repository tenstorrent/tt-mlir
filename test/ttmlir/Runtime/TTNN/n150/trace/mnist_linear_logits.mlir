// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-trace=true" -o mnist_linear_out.mlir %s
// RUN: FileCheck %s --input-file=mnist_linear_out.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer mnist_linear_out.mlir > %t.ttnn
module @MNISTLinear attributes {} {
  func.func @forward(%arg0: tensor<256x7840xbf16>, %arg1: tensor<7840x2560xbf16>, %arg2: tensor<2560xbf16>, %arg3: tensor<2560x100xbf16>, %arg4: tensor<100xbf16>) -> tensor<256x100xbf16> {
    // CHECK: ttnn.trace
    %0 = ttir.empty() : tensor<256x2560xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<256x7840xbf16>, tensor<7840x2560xbf16>, tensor<256x2560xbf16>) -> tensor<256x2560xbf16>
    %2 = ttir.empty() : tensor<256x2560xbf16>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<256x2560xbf16>, tensor<2560xbf16>, tensor<256x2560xbf16>) -> tensor<256x2560xbf16>
    %4 = ttir.empty() : tensor<256x2560xbf16>
    %5 = "ttir.relu"(%3, %4) : (tensor<256x2560xbf16>, tensor<256x2560xbf16>) -> tensor<256x2560xbf16>
    %6 = ttir.empty() : tensor<256x100xbf16>
    %7 = "ttir.matmul"(%5, %arg3, %6) : (tensor<256x2560xbf16>, tensor<2560x100xbf16>, tensor<256x100xbf16>) -> tensor<256x100xbf16>
    %8 = ttir.empty() : tensor<256x100xbf16>
    %9 = "ttir.add"(%7, %arg4, %8) : (tensor<256x100xbf16>, tensor<100xbf16>, tensor<256x100xbf16>) -> tensor<256x100xbf16>
    return %9 : tensor<256x100xbf16>
  }
}
