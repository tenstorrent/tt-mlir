// RUN: ttmlir-opt --stablehlo-complex-math-expander --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t
// REQUIRES: stablehlo

// The Lumina RoPE rotary embedding builds a complex value (view_as_complex),
// upcasts it, multiplies by the complex freqs_cis, then extracts real/imag.
// The complex-math-expander decomposes complex.multiply into real FOIL
// arithmetic:
//   (xr + i*xi) * (yr + i*yi) = (xr*yr - xi*yi) + i*(xr*yi + xi*yr)
// and the complex-data-type-conversion pass then lowers the resulting
// complex/real/imag ops onto the float-pair (...x2) representation.
// No complex type may survive after conversion.

module @complex_rope_multiply {
  // Complex args/result become the trailing real/imag-pair form.
  // CHECK-LABEL: func.func @main
  // CHECK-SAME: %arg0: tensor<1x256x48x2xf64>
  // CHECK-SAME: %arg1: tensor<1x256x12x48xf32>
  // CHECK-SAME: -> tensor<1x256x12x48x2xf64>
  func.func @main(%freqs: tensor<1x256x48xcomplex<f64>>, %re: tensor<1x256x12x48xf32>, %im: tensor<1x256x12x48xf32>) -> tensor<1x256x12x48xcomplex<f64>> {
    // No complex type anywhere in the converted body.
    // CHECK-NOT: complex<
    %c = stablehlo.complex %re, %im : tensor<1x256x12x48xcomplex<f32>>
    %c64 = stablehlo.convert %c : (tensor<1x256x12x48xcomplex<f32>>) -> tensor<1x256x12x48xcomplex<f64>>
    %f = stablehlo.broadcast_in_dim %freqs, dims = [0, 1, 3] : (tensor<1x256x48xcomplex<f64>>) -> tensor<1x256x12x48xcomplex<f64>>
    // The complex multiply lowers to real FOIL arithmetic.
    // CHECK: stablehlo.multiply
    // CHECK: stablehlo.subtract
    // CHECK: stablehlo.multiply
    // CHECK: stablehlo.add
    %prod = stablehlo.multiply %c64, %f : tensor<1x256x12x48xcomplex<f64>>
    return %prod : tensor<1x256x12x48xcomplex<f64>>
  }
}
