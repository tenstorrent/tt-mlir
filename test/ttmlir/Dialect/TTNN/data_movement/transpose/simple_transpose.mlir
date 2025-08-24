// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>) -> tensor<128x64xbf16> {
    %0 = ttir.empty() : tensor<128x64xbf16>
    // CHECK: = "ttnn.permute"
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
    return %1 : tensor<128x64xbf16>
  }
}
