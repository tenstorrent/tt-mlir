// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: = "ttir.arange"
    // CHECK: = "ttir.transpose"
    // CHECK: = "ttir.broadcast"
    %1 = "ttir.arange"() <{start = 0: si64, end = 32: si64, step = 1: si64, arange_dimension = 1: i64}> : () -> tensor<1x32x128x128xf32>
    return %1 : tensor<1x32x128x128xf32>
  }
}
