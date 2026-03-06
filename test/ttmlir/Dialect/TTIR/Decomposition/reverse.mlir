// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @reverse_op(%arg0: tensor<3x3x32x64xf32>) -> tensor<3x3x32x64xf32> {
    // CHECK: %[[V0:[0-9]+]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 0, 3>}> : (tensor<3x3x32x64xf32>) -> tensor<3x32x3x64xf32>
    // CHECK: %[[V1:[0-9]+]] = "ttir.reshape"(%[[V0]]) <{shape = [96 : i32, 192 : i32]}> : (tensor<3x32x3x64xf32>) -> tensor<96x192xf32>
    // CHECK: %[[V2:[0-9]+]] = "ttir.constant"() <{value = dense<[95, 94, 93, {{.*}}, 2, 1, 0]> : tensor<96xsi32>}> : () -> tensor<96xsi32>
    // CHECK: %[[V3:[0-9]+]] = "ttir.embedding"(%[[V2]], %[[V1]]) : (tensor<96xsi32>, tensor<96x192xf32>) -> tensor<96x192xf32>
    // CHECK: %[[V4:[0-9]+]] = "ttir.reshape"(%[[V3]]) <{shape = [3 : i32, 32 : i32, 3 : i32, 64 : i32]}> : (tensor<96x192xf32>) -> tensor<3x32x3x64xf32>
    // CHECK: %[[V5:[0-9]+]] = "ttir.permute"(%[[V4]]) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<3x32x3x64xf32>) -> tensor<3x3x32x64xf32>
    %1 = "ttir.reverse"(%arg0) <{dimensions = array<i64: 1,2>}> : (tensor<3x3x32x64xf32>) -> tensor<3x3x32x64xf32>
    return %1 : tensor<3x3x32x64xf32>
  }
}
