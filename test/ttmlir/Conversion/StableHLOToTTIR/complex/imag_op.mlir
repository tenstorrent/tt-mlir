// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test_imag(%arg0: tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64> {
    %0 = "stablehlo.imag"(%arg0) : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64>
    // CHECK: "ttir.permute"
    // CHECK-SAME: {permutation = array<i64: 2, 0, 1>}
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [1 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [2 : i32, 2 : i32, 4 : i32]
    // CHECK: "ttir.reshape"({{.*}}) {{.*}} -> tensor<2x4xf64>
    return %0 : tensor<2x4xf64>
  }
}
