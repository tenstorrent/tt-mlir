// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test_real(%arg0: tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64> {
    %0 = "stablehlo.real"(%arg0) : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64>
    // CHECK: "ttir.slice_static"(%arg0)
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [2 : i32, 4 : i32, 1 : i32]
    // CHECK: "ttir.reshape"({{.*}}) {{.*}} -> tensor<2x4xf64>
    return %0 : tensor<2x4xf64>
  }
}
