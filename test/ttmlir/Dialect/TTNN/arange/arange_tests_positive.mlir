// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @arange(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: = "ttnn.arange"
    %0 = "ttir.arange"() <{start = 0: si64, end = 32: si64, step = 1: si64, arange_dimension = 1: i64}> : () -> tensor<1x32x128x128xf32>
    %1 = "ttir.multiply"(%arg0, %0) : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %1 : tensor<1x32x128x128xf32>
  }

  func.func @arange_multiple_users() -> (tensor<1x1024x1xi32>, tensor<1x1x1024xi32>) {
    %0 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 1024 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1024xi32>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 1024 : i32, 1 : i32]}> : (tensor<1024xi32>) -> tensor<1x1024x1xi32>
    %2 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xi32>) -> tensor<1x1x1024xi32>
    // Verify that the arange is op in IR is before both reshape ops
    // CHECK: "ttnn.arange"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reshape"
    return %1, %2: tensor<1x1024x1xi32>, tensor<1x1x1024xi32>
  }
}
