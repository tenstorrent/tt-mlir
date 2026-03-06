// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @reverse_op(%arg0: tensor<3x3x32x64xf32>) -> tensor<3x3x32x64xf32> {
    // CHECK: "ttir.permute"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.constant"
    // CHECK: "ttir.embedding"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.permute"
    %1 = "ttir.reverse"(%arg0) <{dimensions = array<i64: 1,2>}> : (tensor<3x3x32x64xf32>) -> tensor<3x3x32x64xf32>
    return %1 : tensor<3x3x32x64xf32>
  }
}
