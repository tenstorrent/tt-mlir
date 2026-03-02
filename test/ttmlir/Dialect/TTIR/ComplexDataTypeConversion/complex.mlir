// RUN: ttmlir-opt --ttir-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_complex_0(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>)
    -> (tensor<2x4xcomplex<f32>>) {
        // CHECK: "ttir.complex"
        // CHECK: (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4x2xf32>
        %0 = "ttir.stablehlo_complex"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xcomplex<f32>>
        return %0 : tensor<2x4xcomplex<f32>>
    }
    func.func @test_complex_1(%arg0: tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32> {
        // CHECK: "ttir.real"
        // CHECK: (tensor<2x4x2xf32>) -> tensor<2x4xf32>
        %0 = "ttir.stablehlo_real"(%arg0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        return %0 : tensor<2x4xf32>
    }
    func.func @test_complex_2(%arg0: tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32> {
        // CHECK: "ttir.imag"
        // CHECK: (tensor<2x4x2xf32>) -> tensor<2x4xf32>
        %0 = "ttir.stablehlo_imag"(%arg0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        return %0 : tensor<2x4xf32>
    }
    func.func @test_complex(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>)
    -> (tensor<2x4xf32>, tensor<2x4xf32>) {
        // CHECK: "ttir.complex"
        // CHECK: (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4x2xf32>
        %0 = "ttir.stablehlo_complex"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xcomplex<f32>>
        // CHECK: "ttir.real"
        // CHECK: (tensor<2x4x2xf32>) -> tensor<2x4xf32>
        %1 = "ttir.stablehlo_real"(%0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        // CHECK: "ttir.imag"
        // CHECK: (tensor<2x4x2xf32>) -> tensor<2x4xf32>
        %2 = "ttir.stablehlo_imag"(%0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        return %1, %2 : tensor<2x4xf32>, tensor<2x4xf32>
    }
}
