// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

// stablehlo.convert between complex element types (e.g. the complex<f32> ->
// complex<f64> upcast emitted for rotary positional embeddings) is a dtype
// cast, not arithmetic, so the complex-math-expander never touches it. This
// pass must unpack it to a convert on the trailing real/imag-pair layout,
// otherwise the surrounding unpacked ops leave a live complex materialization.
module {
    func.func @test_complex_convert_upcast(%arg0: tensor<4x8xcomplex<f32>>) -> tensor<4x8xcomplex<f64>> {
        // CHECK-LABEL: func.func @test_complex_convert_upcast
        // CHECK-SAME: tensor<4x8x2xf32>
        // CHECK-SAME: -> tensor<4x8x2xf64>
        // CHECK: stablehlo.convert
        // CHECK-SAME: (tensor<4x8x2xf32>) -> tensor<4x8x2xf64>
        // CHECK-NOT: complex
        %0 = stablehlo.convert %arg0 : (tensor<4x8xcomplex<f32>>) -> tensor<4x8xcomplex<f64>>
        return %0 : tensor<4x8xcomplex<f64>>
    }

    func.func @test_complex_convert_downcast(%arg0: tensor<2x3x16xcomplex<f64>>) -> tensor<2x3x16xcomplex<f32>> {
        // CHECK-LABEL: func.func @test_complex_convert_downcast
        // CHECK-SAME: tensor<2x3x16x2xf64>
        // CHECK-SAME: -> tensor<2x3x16x2xf32>
        // CHECK: stablehlo.convert
        // CHECK-SAME: (tensor<2x3x16x2xf64>) -> tensor<2x3x16x2xf32>
        // CHECK-NOT: complex
        %0 = stablehlo.convert %arg0 : (tensor<2x3x16xcomplex<f64>>) -> tensor<2x3x16xcomplex<f32>>
        return %0 : tensor<2x3x16xcomplex<f32>>
    }
}
