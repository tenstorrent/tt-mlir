// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

// Shardy wraps reshard collectives in stablehlo.composite ops (e.g.
// "sdy.all_slice") whose result carries complex freqs_cis. The composite result
// must be converted to the unpacked real (...x2xf32) representation in lock-step
// with its decomposition function's signature (tt-xla #5313); otherwise the
// complex result feeds a converted consumer and leaves an unresolved
// materialization.
module {
    // CHECK-LABEL: func.func @test_complex_composite
    // CHECK-SAME: tensor<42x64x2xf32>
    func.func @test_complex_composite(%arg0: tensor<42x64xcomplex<f32>>) -> tensor<1x42x16xcomplex<f32>> {
        // CHECK: stablehlo.composite
        // CHECK-SAME: decomposition = @sdy.all_slice
        // CHECK-SAME: (tensor<42x64x2xf32>) -> tensor<42x16x2xf32>
        %0 = stablehlo.composite "sdy.all_slice" %arg0 {decomposition = @sdy.all_slice} : (tensor<42x64xcomplex<f32>>) -> tensor<42x16xcomplex<f32>>
        // CHECK: stablehlo.reshape
        // CHECK-SAME: (tensor<42x16x2xf32>) -> tensor<1x42x16x2xf32>
        %1 = stablehlo.reshape %0 : (tensor<42x16xcomplex<f32>>) -> tensor<1x42x16xcomplex<f32>>
        return %1 : tensor<1x42x16xcomplex<f32>>
    }

    // CHECK: func.func private @sdy.all_slice
    // CHECK-SAME: tensor<42x64x2xf32>
    // CHECK-SAME: -> tensor<42x16x2xf32>
    func.func private @sdy.all_slice(%arg0: tensor<42x64xcomplex<f32>>) -> tensor<42x16xcomplex<f32>> {
        %0 = stablehlo.slice %arg0 [0:42, 0:16] : (tensor<42x64xcomplex<f32>>) -> tensor<42x16xcomplex<f32>>
        return %0 : tensor<42x16xcomplex<f32>>
    }
}
