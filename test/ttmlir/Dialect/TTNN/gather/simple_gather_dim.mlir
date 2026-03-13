// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @forward(%arg0: tensor<32x64xbf16>, %arg1: tensor<32x16xui32>) -> tensor<32x16xbf16> {
    // CHECK: "ttnn.gather"
    %1 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 1 : i32}> : (tensor<32x64xbf16>, tensor<32x16xui32>) -> tensor<32x16xbf16>
    return %1 : tensor<32x16xbf16>
  }
}
