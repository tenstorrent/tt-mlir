// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion %s

// This test verifies that complex types in function arguments, reshapes, and
// real/imag extraction are correctly decomposed to float-pair representations.
func.func @main(%arg0: tensor<16x8xcomplex<f32>>, %arg1: tensor<2x16x4x16xbf16>) -> (tensor<2x16x4x16xbf16>) {
    %0 = stablehlo.convert %arg1 : (tensor<2x16x4x16xbf16>) -> tensor<2x16x4x16xf32>
    %1 = stablehlo.reshape %0 : (tensor<2x16x4x16xf32>) -> tensor<2x16x4x8x2xf32>
    %2 = stablehlo.slice %1 [0:2, 0:16, 0:4, 0:8, 0:1] : (tensor<2x16x4x8x2xf32>) -> tensor<2x16x4x8x1xf32>
    %3 = stablehlo.slice %1 [0:2, 0:16, 0:4, 0:8, 1:2] : (tensor<2x16x4x8x2xf32>) -> tensor<2x16x4x8x1xf32>
    %4 = stablehlo.reshape %2 : (tensor<2x16x4x8x1xf32>) -> tensor<2x16x4x8xf32>
    %5 = stablehlo.reshape %arg0 : (tensor<16x8xcomplex<f32>>) -> tensor<1x16x8xcomplex<f32>>
    %6 = stablehlo.reshape %5 : (tensor<1x16x8xcomplex<f32>>) -> tensor<1x16x1x8xcomplex<f32>>
    %7 = stablehlo.real %6 : (tensor<1x16x1x8xcomplex<f32>>) -> tensor<1x16x1x8xf32>
    %8 = stablehlo.reshape %7 : (tensor<1x16x1x8xf32>) -> tensor<16x8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [1, 3] : (tensor<16x8xf32>) -> tensor<2x16x4x8xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<2x16x4x8xf32>
    %11 = stablehlo.reshape %3 : (tensor<2x16x4x8x1xf32>) -> tensor<2x16x4x8xf32>
    %12 = stablehlo.imag %6 : (tensor<1x16x1x8xcomplex<f32>>) -> tensor<1x16x1x8xf32>
    %13 = stablehlo.reshape %12 : (tensor<1x16x1x8xf32>) -> tensor<16x8xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [1, 3] : (tensor<16x8xf32>) -> tensor<2x16x4x8xf32>
    %15 = stablehlo.multiply %11, %14 : tensor<2x16x4x8xf32>
    %16 = stablehlo.subtract %10, %15 : tensor<2x16x4x8xf32>
    %17 = stablehlo.multiply %4, %14 : tensor<2x16x4x8xf32>
    %18 = stablehlo.multiply %11, %9 : tensor<2x16x4x8xf32>
    %19 = stablehlo.add %17, %18 : tensor<2x16x4x8xf32>
    %20 = stablehlo.reshape %16 : (tensor<2x16x4x8xf32>) -> tensor<2x16x4x8x1xf32>
    %21 = stablehlo.reshape %19 : (tensor<2x16x4x8xf32>) -> tensor<2x16x4x8x1xf32>
    %22 = stablehlo.concatenate %20, %21, dim = 4 : (tensor<2x16x4x8x1xf32>, tensor<2x16x4x8x1xf32>) -> tensor<2x16x4x8x2xf32>
    %23 = stablehlo.reshape %22 : (tensor<2x16x4x8x2xf32>) -> tensor<2x16x4x16xf32>
    %24 = stablehlo.convert %23 : (tensor<2x16x4x16xf32>) -> tensor<2x16x4x16xbf16>
    return %24 : tensor<2x16x4x16xbf16>
}
