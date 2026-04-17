// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_complex_concat_dim0(%arg0: tensor<4x8xcomplex<f32>>, %arg1: tensor<4x8xcomplex<f32>>)
    -> tensor<8x8xcomplex<f32>> {
        // CHECK: stablehlo.concatenate
        // CHECK-SAME: dim = 0
        // CHECK-SAME: (tensor<4x8x2xf32>, tensor<4x8x2xf32>) -> tensor<8x8x2xf32>
        %0 = "stablehlo.concatenate"(%arg0, %arg1) {
            dimension = 0 : i64
        } : (tensor<4x8xcomplex<f32>>, tensor<4x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>>
        return %0 : tensor<8x8xcomplex<f32>>
    }

    func.func @test_complex_concat_dim1(%arg0: tensor<4x8xcomplex<f32>>, %arg1: tensor<4x8xcomplex<f32>>)
    -> tensor<4x16xcomplex<f32>> {
        // CHECK: stablehlo.concatenate
        // CHECK-SAME: dim = 1
        // CHECK-SAME: (tensor<4x8x2xf32>, tensor<4x8x2xf32>) -> tensor<4x16x2xf32>
        %0 = "stablehlo.concatenate"(%arg0, %arg1) {
            dimension = 1 : i64
        } : (tensor<4x8xcomplex<f32>>, tensor<4x8xcomplex<f32>>) -> tensor<4x16xcomplex<f32>>
        return %0 : tensor<4x16xcomplex<f32>>
    }

    func.func @test_complex_concat_three(%arg0: tensor<2x4xcomplex<f32>>, %arg1: tensor<2x4xcomplex<f32>>,
                                          %arg2: tensor<2x4xcomplex<f32>>)
    -> tensor<6x4xcomplex<f32>> {
        // CHECK: stablehlo.concatenate
        // CHECK-SAME: dim = 0
        // CHECK-SAME: (tensor<2x4x2xf32>, tensor<2x4x2xf32>, tensor<2x4x2xf32>) -> tensor<6x4x2xf32>
        %0 = "stablehlo.concatenate"(%arg0, %arg1, %arg2) {
            dimension = 0 : i64
        } : (tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>) -> tensor<6x4xcomplex<f32>>
        return %0 : tensor<6x4xcomplex<f32>>
    }
}
