// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module attributes {} {
  func.func @arange_dim2(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK: = "ttnn.arange"
    %0 = "ttir.arange"() <{start = 0: si64, end = 128: si64, step = 1: si64, arange_dimension = 3: i64}> : () -> tensor<1x1x32x128xbf16>
    %1 = ttir.empty() : tensor<1x1x32x128xbf16>
    %2 = "ttir.multiply"(%arg0, %0, %1) : (tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    return %2 : tensor<1x1x32x128xbf16>
  }

  func.func @arange_dim3(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK: = "ttnn.arange"
    %0 = "ttir.arange"() <{start = 0: si64, end = 64: si64, step = 2: si64, arange_dimension = 2: i64}> : () -> tensor<1x1x32x128xbf16>
    %1 = ttir.empty() : tensor<1x1x32x128xbf16>
    %2 = "ttir.multiply"(%arg0, %0, %1) : (tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    return %2 : tensor<1x1x32x128xbf16>
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
    // CHECK: "ttnn.reshape"
    return %2, %4: tensor<1x1024x1xi32>, tensor<1x1x1024xi32>
  }
}
