// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test_complex(%arg0: tensor<2x4xf64>, %arg1: tensor<2x4xf64>) -> tensor<2x4xcomplex<f64>> {
    %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x4xcomplex<f64>>
    // CHECK: "ttir.reshape"(%arg0) {{.*}} -> tensor<1x2x4xf64>
    // CHECK: "ttir.reshape"(%arg1) {{.*}} -> tensor<1x2x4xf64>
    // CHECK: "ttir.concat"({{.*}}) {{.*}} -> tensor<2x2x4xf64>
    // CHECK: "ttir.permute"({{.*}}) {{.*}} -> tensor<2x4x2xf64>
    return %0 : tensor<2x4xcomplex<f64>>
  }
}
