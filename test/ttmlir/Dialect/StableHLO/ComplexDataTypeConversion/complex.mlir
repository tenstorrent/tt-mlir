// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_complex_0(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>)
    -> (tensor<2x4xcomplex<f32>>) {
        // CHECK: stablehlo.reshape
        // CHECK-SAME: (tensor<2x4xf32>) -> tensor<2x4x1xf32>
        // CHECK: stablehlo.reshape
        // CHECK-SAME: (tensor<2x4xf32>) -> tensor<2x4x1xf32>
        // CHECK: stablehlo.concatenate
        // CHECK-SAME: (tensor<2x4x1xf32>, tensor<2x4x1xf32>) -> tensor<2x4x2xf32>
        %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xcomplex<f32>>
        return %0 : tensor<2x4xcomplex<f32>>
    }
    func.func @test_complex_1(%arg0: tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32> {
        // CHECK: stablehlo.slice
        // CHECK-SAME: [0:2, 0:4, 0:1]
        // CHECK: stablehlo.reshape
        // CHECK-SAME: (tensor<2x4x1xf32>) -> tensor<2x4xf32>
        %0 = "stablehlo.real"(%arg0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        return %0 : tensor<2x4xf32>
    }
    func.func @test_complex_2(%arg0: tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32> {
        // CHECK: stablehlo.slice
        // CHECK-SAME: [0:2, 0:4, 1:2]
        // CHECK: stablehlo.reshape
        // CHECK-SAME: (tensor<2x4x1xf32>) -> tensor<2x4xf32>
        %0 = "stablehlo.imag"(%arg0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        return %0 : tensor<2x4xf32>
    }
    func.func @test_complex(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>)
    -> (tensor<2x4xf32>, tensor<2x4xf32>) {
        // CHECK: stablehlo.reshape
        // CHECK: stablehlo.reshape
        // CHECK: stablehlo.concatenate
        %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xcomplex<f32>>
        // CHECK: stablehlo.slice
        // CHECK: stablehlo.reshape
        %1 = "stablehlo.real"(%0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        // CHECK: stablehlo.slice
        // CHECK: stablehlo.reshape
        %2 = "stablehlo.imag"(%0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        return %1, %2 : tensor<2x4xf32>, tensor<2x4xf32>
    }
}
