// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @reverse_op(%arg0: tensor<3x3x32x64xf32>) -> tensor<3x3x32x64xf32> {
    %0 = ttir.empty() : tensor<3x3x32x64xf32>
    // CHECK: "ttir.constant"() <{value = dense<[2, 1, 0]>
    // CHECK: "ttir.embedding"
    // CHECK: "ttir.constant"() <{value = dense<[31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]>
    // CHECK: "ttir.embedding"
    %1 = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64: 1,2>}> : (tensor<3x3x32x64xf32>, tensor<3x3x32x64xf32>) -> tensor<3x3x32x64xf32>
    return %1 : tensor<3x3x32x64xf32>
  }
}
