// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

// stablehlo.select on complex operands: the true/false values unpack to the
// trailing real/imag-pair layout, and the i1 predicate must gain the same
// trailing dim so its shape still matches the operands.
module {
    func.func @test_complex_select(%pred: tensor<4x8xi1>, %on_true: tensor<4x8xcomplex<f32>>, %on_false: tensor<4x8xcomplex<f32>>) -> tensor<4x8xcomplex<f32>> {
        // CHECK-LABEL: func.func @test_complex_select
        // CHECK-SAME: tensor<4x8xi1>
        // CHECK-SAME: tensor<4x8x2xf32>
        // CHECK-SAME: tensor<4x8x2xf32>
        // CHECK-SAME: -> tensor<4x8x2xf32>
        // Predicate is broadcast to the new trailing real/imag dim.
        // CHECK: stablehlo.broadcast_in_dim
        // CHECK-SAME: dims = [0, 1]
        // CHECK-SAME: (tensor<4x8xi1>) -> tensor<4x8x2xi1>
        // CHECK: stablehlo.select
        // CHECK-SAME: tensor<4x8x2xi1>, tensor<4x8x2xf32>
        // CHECK-NOT: complex
        %0 = stablehlo.select %pred, %on_true, %on_false : tensor<4x8xi1>, tensor<4x8xcomplex<f32>>
        return %0 : tensor<4x8xcomplex<f32>>
    }
}
