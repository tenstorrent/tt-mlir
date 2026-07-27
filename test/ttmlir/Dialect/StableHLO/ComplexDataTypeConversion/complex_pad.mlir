// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

// stablehlo.pad on complex tensors (e.g. padding rotary-embedding freqs_cis to
// the max sequence length). Padding is defined over the complex dims and the
// pad value is a complex scalar constant, so the real/imag planes are padded
// separately with their own scalar constant pad value (extracted from the
// original complex constant) and re-interleaved. The pad value must stay a
// direct constant so downstream StableHLOToTTIR PadOp lowering can read it.
module {
    func.func @test_complex_pad(%arg0: tensor<2x3xcomplex<f32>>) -> tensor<4x3xcomplex<f32>> {
        // CHECK-LABEL: func.func @test_complex_pad
        // CHECK-SAME: tensor<2x3x2xf32>
        // CHECK-SAME: -> tensor<4x3x2xf32>
        // Real and imag planes are each padded with a rank-0 scalar constant.
        // CHECK-DAG: %[[RE:.*]] = stablehlo.constant {{.*}} tensor<f32>
        // CHECK-DAG: %[[IM:.*]] = stablehlo.constant {{.*}} tensor<f32>
        // CHECK: stablehlo.pad
        // CHECK-SAME: low = [0, 0]
        // CHECK-SAME: high = [2, 0]
        // CHECK-SAME: (tensor<2x3xf32>, tensor<f32>) -> tensor<4x3xf32>
        // CHECK: stablehlo.pad
        // CHECK-SAME: (tensor<2x3xf32>, tensor<f32>) -> tensor<4x3xf32>
        // CHECK-NOT: complex
        %pad = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
        %0 = stablehlo.pad %arg0, %pad, low = [0, 0], high = [2, 0], interior = [0, 0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x3xcomplex<f32>>
        return %0 : tensor<4x3xcomplex<f32>>
    }
}
