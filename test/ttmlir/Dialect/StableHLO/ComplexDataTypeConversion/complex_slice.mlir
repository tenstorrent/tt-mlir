// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_complex_slice(%arg0: tensor<4x8xcomplex<f32>>) -> tensor<2x4xcomplex<f32>> {
        // CHECK: stablehlo.slice
        // CHECK-SAME: [0:2, 0:4, 0:2]
        // CHECK-SAME: (tensor<4x8x2xf32>) -> tensor<2x4x2xf32>
        %0 = "stablehlo.slice"(%arg0) {
            start_indices = array<i64: 0, 0>,
            limit_indices = array<i64: 2, 4>,
            strides = array<i64: 1, 1>
        } : (tensor<4x8xcomplex<f32>>) -> tensor<2x4xcomplex<f32>>
        return %0 : tensor<2x4xcomplex<f32>>
    }

    func.func @test_complex_slice_with_strides(%arg0: tensor<8x16xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> {
        // CHECK: stablehlo.slice
        // CHECK-SAME: [0:8:2, 0:16:4, 0:2]
        // CHECK-SAME: (tensor<8x16x2xf32>) -> tensor<4x4x2xf32>
        %0 = "stablehlo.slice"(%arg0) {
            start_indices = array<i64: 0, 0>,
            limit_indices = array<i64: 8, 16>,
            strides = array<i64: 2, 4>
        } : (tensor<8x16xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
        return %0 : tensor<4x4xcomplex<f32>>
    }

    func.func @test_complex_slice_with_offset(%arg0: tensor<64x32xcomplex<f32>>) -> tensor<64x28xcomplex<f32>> {
        // CHECK: stablehlo.slice
        // CHECK-SAME: [0:64, 0:28, 0:2]
        // CHECK-SAME: (tensor<64x32x2xf32>) -> tensor<64x28x2xf32>
        %0 = "stablehlo.slice"(%arg0) {
            start_indices = array<i64: 0, 0>,
            limit_indices = array<i64: 64, 28>,
            strides = array<i64: 1, 1>
        } : (tensor<64x32xcomplex<f32>>) -> tensor<64x28xcomplex<f32>>
        return %0 : tensor<64x28xcomplex<f32>>
    }
}
