// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_complex(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>)
    -> (tensor<2x4xcomplex<f32>>) {
        // CHECK: stablehlo.reshape
        // CHECK-SAME: (tensor<2x4xf32>) -> tensor<1x2x4xf32>
        // CHECK: stablehlo.reshape
        // CHECK-SAME: (tensor<2x4xf32>) -> tensor<1x2x4xf32>
        // CHECK: stablehlo.concatenate
        // CHECK-SAME: (tensor<1x2x4xf32>, tensor<1x2x4xf32>) -> tensor<2x2x4xf32>
        // CHECK: stablehlo.transpose
        // CHECK-SAME: (tensor<2x2x4xf32>) -> tensor<2x4x2xf32>
        %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xcomplex<f32>>
        return %0 : tensor<2x4xcomplex<f32>>
    }
    func.func @test_real(%arg0: tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32> {
        // CHECK: stablehlo.transpose
        // CHECK-SAME: (tensor<2x4x2xf32>) -> tensor<2x2x4xf32>
        // CHECK: stablehlo.slice
        // CHECK-SAME: [0:1, 0:2, 0:4]
        // CHECK: stablehlo.reshape
        // CHECK-SAME: (tensor<1x2x4xf32>) -> tensor<2x4xf32>
        %0 = "stablehlo.real"(%arg0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        return %0 : tensor<2x4xf32>
    }
    func.func @test_imag(%arg0: tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32> {
        // CHECK: stablehlo.transpose
        // CHECK-SAME: (tensor<2x4x2xf32>) -> tensor<2x2x4xf32>
        // CHECK: stablehlo.slice
        // CHECK-SAME: [1:2, 0:2, 0:4]
        // CHECK: stablehlo.reshape
        // CHECK-SAME: (tensor<1x2x4xf32>) -> tensor<2x4xf32>
        %0 = "stablehlo.imag"(%arg0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        return %0 : tensor<2x4xf32>
    }
    func.func @test_complex_chain(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>)
    -> (tensor<2x4xf32>, tensor<2x4xf32>) {
        // CHECK: stablehlo.reshape
        // CHECK: stablehlo.reshape
        // CHECK: stablehlo.concatenate
        // CHECK: stablehlo.transpose
        // CHECK-SAME: (tensor<2x2x4xf32>) -> tensor<2x4x2xf32>
        %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xcomplex<f32>>
        // CHECK: stablehlo.transpose
        // CHECK-SAME: (tensor<2x4x2xf32>) -> tensor<2x2x4xf32>
        // CHECK: stablehlo.slice
        // CHECK: stablehlo.reshape
        %1 = "stablehlo.real"(%0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        // CHECK: stablehlo.transpose
        // CHECK-SAME: (tensor<2x4x2xf32>) -> tensor<2x2x4xf32>
        // CHECK: stablehlo.slice
        // CHECK: stablehlo.reshape
        %2 = "stablehlo.imag"(%0) : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xf32>
        return %1, %2 : tensor<2x4xf32>, tensor<2x4xf32>
    }
    func.func @test_complex_broadcast_in_dim(%arg0: tensor<2x4xcomplex<f32>>) -> tensor<3x2x4xcomplex<f32>> {
        // CHECK: stablehlo.broadcast_in_dim
        // CHECK-SAME: dims = [1, 2, 3]
        // CHECK-SAME: (tensor<2x4x2xf32>) -> tensor<3x2x4x2xf32>
        %0 = "stablehlo.broadcast_in_dim"(%arg0) {
            broadcast_dimensions = array<i64: 1, 2>
        } : (tensor<2x4xcomplex<f32>>) -> tensor<3x2x4xcomplex<f32>>
        return %0 : tensor<3x2x4xcomplex<f32>>
    }

    func.func @test_complex_reshape(%arg0: tensor<2x4xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
        // CHECK: stablehlo.reshape
        // CHECK-SAME: (tensor<2x4x2xf32>) -> tensor<8x2xf32>
        %0 = "stablehlo.reshape"(%arg0) : (tensor<2x4xcomplex<f32>>) -> tensor<8xcomplex<f32>>
        return %0 : tensor<8xcomplex<f32>>
    }
}
