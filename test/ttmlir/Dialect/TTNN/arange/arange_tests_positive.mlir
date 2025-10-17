// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @arange(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: = "ttnn.arange"
    %1 = "ttir.arange"() <{start = 0: si64, end = 32: si64, step = 1: si64, arange_dimension = 1: i64}> : () -> tensor<1x32x128x128xf32>
    %dps = ttir.empty() : tensor<1x32x128x128xf32>
    %2 = "ttir.multiply"(%arg0, %1, %dps) : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %2 : tensor<1x32x128x128xf32>
  }

  func.func @arange_multiple_users() -> (tensor<1x1024x1xi32>, tensor<1x1x1024xi32>) {
    %0 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 1024 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1024xi32>
    %1 = ttir.empty() : tensor<1x1024x1xi32>
    %2 = "ttir.reshape"(%0, %1) <{shape = [1 : i32, 1024 : i32, 1 : i32]}> : (tensor<1024xi32>, tensor<1x1024x1xi32>) -> tensor<1x1024x1xi32>
    %3 = ttir.empty() : tensor<1x1x1024xi32>
    %4 = "ttir.reshape"(%0, %3) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xi32>, tensor<1x1x1024xi32>) -> tensor<1x1x1024xi32>
    // Verify that the arange is op in IR is before both reshape ops
    // CHECK: "ttnn.arange"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.view"
    return %2, %4: tensor<1x1024x1xi32>, tensor<1x1x1024xi32>
  }
}
