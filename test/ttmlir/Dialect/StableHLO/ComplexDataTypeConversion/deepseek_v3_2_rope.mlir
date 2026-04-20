// RUN: ttmlir-opt --stablehlo-complex-math-expander --stablehlo-complex-data-type-conversion %s
// REQUIRES: stablehlo

func.func @main(%arg0: tensor<16x8xcomplex<f32>>, %arg1: tensor<2x16x4x16xbf16>) -> tensor<2x16x4x16xbf16> {
  %0 = stablehlo.convert %arg1 : (tensor<2x16x4x16xbf16>) -> tensor<2x16x4x16xf32>
  %1 = stablehlo.reshape %0 : (tensor<2x16x4x16xf32>) -> tensor<2x16x4x8x2xf32>
  %2 = stablehlo.slice %1 [0:2, 0:16, 0:4, 0:8, 0:1] : (tensor<2x16x4x8x2xf32>) -> tensor<2x16x4x8x1xf32>
  %3 = stablehlo.reshape %2 : (tensor<2x16x4x8x1xf32>) -> tensor<2x16x4x8xf32>
  %4 = stablehlo.slice %1 [0:2, 0:16, 0:4, 0:8, 1:2] : (tensor<2x16x4x8x2xf32>) -> tensor<2x16x4x8x1xf32>
  %5 = stablehlo.reshape %4 : (tensor<2x16x4x8x1xf32>) -> tensor<2x16x4x8xf32>
  %6 = stablehlo.complex %3, %5 : tensor<2x16x4x8xcomplex<f32>>
  %7 = stablehlo.reshape %arg0 : (tensor<16x8xcomplex<f32>>) -> tensor<1x16x8xcomplex<f32>>
  %8 = stablehlo.reshape %7 : (tensor<1x16x8xcomplex<f32>>) -> tensor<16x8xcomplex<f32>>
  %9 = stablehlo.broadcast_in_dim %8, dims = [1, 3] : (tensor<16x8xcomplex<f32>>) -> tensor<2x16x4x8xcomplex<f32>>
  %10 = stablehlo.multiply %6, %9 : tensor<2x16x4x8xcomplex<f32>>
  %11 = stablehlo.real %10 : (tensor<2x16x4x8xcomplex<f32>>) -> tensor<2x16x4x8xf32>
  %12 = stablehlo.reshape %11 : (tensor<2x16x4x8xf32>) -> tensor<2x16x4x8x1xf32>
  %13 = stablehlo.imag %10 : (tensor<2x16x4x8xcomplex<f32>>) -> tensor<2x16x4x8xf32>
  %14 = stablehlo.reshape %13 : (tensor<2x16x4x8xf32>) -> tensor<2x16x4x8x1xf32>
  %15 = stablehlo.concatenate %12, %14, dim = 4 : (tensor<2x16x4x8x1xf32>, tensor<2x16x4x8x1xf32>) -> tensor<2x16x4x8x2xf32>
  %16 = stablehlo.reshape %15 : (tensor<2x16x4x8x2xf32>) -> tensor<2x16x4x16xf32>
  %17 = stablehlo.convert %16 : (tensor<2x16x4x16xf32>) -> tensor<2x16x4x16xbf16>
  return %17 : tensor<2x16x4x16xbf16>
}
