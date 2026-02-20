// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test_real(%arg0: tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64> {
    %0 = "stablehlo.real"(%arg0) : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64>
    // CHECK: "ttir.stablehlo_real"(%arg0) {{.*}} -> tensor<2x4xf64>
    return %0 : tensor<2x4xf64>
  }
}
