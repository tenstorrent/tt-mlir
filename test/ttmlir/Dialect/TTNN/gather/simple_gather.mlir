// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  // Test: Basic gather along dim 0
  func.func @gather_dim0(%arg0: tensor<3x4xbf16>, %arg1: tensor<2x4xui32>) -> tensor<2x4xbf16> {
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 0 : i32}> : (tensor<3x4xbf16>, tensor<2x4xui32>) -> tensor<2x4xbf16>
    // CHECK: %{{[0-9]+}} = "ttnn.gather"({{.*}}) <{dim = 0 : i32}> : (tensor<3x4xbf16, {{.*}}>, tensor<2x4xui32, {{.*}}>) -> tensor<2x4xbf16, {{.*}}>
    return %0 : tensor<2x4xbf16>
  }

  // Test: Gather along dim 1
  func.func @gather_dim1(%arg0: tensor<3x4xf32>, %arg1: tensor<3x2xui32>) -> tensor<3x2xf32> {
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 1 : i32}> : (tensor<3x4xf32>, tensor<3x2xui32>) -> tensor<3x2xf32>
    // CHECK: %{{[0-9]+}} = "ttnn.gather"({{.*}}) <{dim = 1 : i32}> : (tensor<3x4xf32, {{.*}}>, tensor<3x2xui32, {{.*}}>) -> tensor<3x2xf32, {{.*}}>
    return %0 : tensor<3x2xf32>
  }

  // Test: Gather on higher-rank tensor
  func.func @gather_4d(%arg0: tensor<2x3x4x5xbf16>, %arg1: tensor<2x3x2x5xui32>) -> tensor<2x3x2x5xbf16> {
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 2 : i32}> : (tensor<2x3x4x5xbf16>, tensor<2x3x2x5xui32>) -> tensor<2x3x2x5xbf16>
    // CHECK: %{{[0-9]+}} = "ttnn.gather"({{.*}}) <{dim = 2 : i32}> : (tensor<2x3x4x5xbf16, {{.*}}>, tensor<2x3x2x5xui32, {{.*}}>) -> tensor<2x3x2x5xbf16, {{.*}}>
    return %0 : tensor<2x3x2x5xbf16>
  }
}
