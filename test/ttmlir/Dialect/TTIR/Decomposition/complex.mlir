// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_complex_0(%arg0: tensor<2x4xf64>, %arg1: tensor<2x4xf64>)
    -> (tensor<2x4xcomplex<f64>>) {
        // CHECK: "ttir.complex"
        // CHECK: (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x4x2xf64>
        %0 = "ttir.stablehlo_complex"(%arg0, %arg1) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x4xcomplex<f64>>
        return %0 : tensor<2x4xcomplex<f64>>
    }
    func.func @test_complex_1(%arg0: tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64> {
        // CHECK: "ttir.real"
        // CHECK: (tensor<2x4x2xf64>) -> tensor<2x4xf64>
        %0 = "ttir.stablehlo_real"(%arg0) : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64>
        return %0 : tensor<2x4xf64>
    }
    func.func @test_complex_2(%arg0: tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64> {
        // CHECK: "ttir.imag"
        // CHECK: (tensor<2x4x2xf64>) -> tensor<2x4xf64>
        %0 = "ttir.stablehlo_imag"(%arg0) : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64>
        return %0 : tensor<2x4xf64>
    }
    func.func @test_complex(%arg0: tensor<2x4xf64>, %arg1: tensor<2x4xf64>)
    -> (tensor<2x4xf64>, tensor<2x4xf64>) {
        // CHECK: "ttir.complex"
        // CHECK: (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x4x2xf64>
        %0 = "ttir.stablehlo_complex"(%arg0, %arg1) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x4xcomplex<f64>>
        // CHECK: "ttir.real"
        // CHECK: (tensor<2x4x2xf64>) -> tensor<2x4xf64>
        %1 = "ttir.stablehlo_real"(%0) : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64>
        // CHECK: "ttir.imag"
        // CHECK: (tensor<2x4x2xf64>) -> tensor<2x4xf64>
        %2 = "ttir.stablehlo_imag"(%0) : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xf64>
        return %1, %2 : tensor<2x4xf64>, tensor<2x4xf64>
    }
}
