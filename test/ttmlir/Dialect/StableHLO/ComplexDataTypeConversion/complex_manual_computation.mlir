// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --stablehlo-complex-math-expander --stablehlo-complex-data-type-conversion %s | FileCheck %s
// REQUIRES: stablehlo

// Test that ComplexDataTypeConversion handles complex types inside
// sdy.manual_computation regions. This is the pattern produced by
// complex RoPE (view_as_complex * polar) with SPMD sharding enabled.

sdy.mesh @mesh = <["_axis_0_aux"=1, "_axis_0"=4]>

// CHECK-LABEL: func.func @test_complex_in_manual_computation
// CHECK-SAME: %arg0: tensor<16x16x2xf32>
// CHECK-SAME: %arg1: tensor<128x128xbf16>
// CHECK-SAME: %arg2: tensor<1x16x128xbf16>
func.func @test_complex_in_manual_computation(
    %arg0: tensor<16x16xcomplex<f32>>,
    %arg1: tensor<128x128xbf16>,
    %arg2: tensor<1x16x128xbf16>
) -> tensor<1x16x128xbf16> {
  // Verify sharding for the complex arg gains a third (trailing) dimension:
  // CHECK: sdy.manual_computation
  // CHECK-SAME: in_shardings=[<@mesh, [{}, {}, {}]>, <@mesh, [{"_axis_0"}, {}]>, <@mesh, [{}, {}, {}]>]
  // CHECK-SAME: out_shardings=[<@mesh, [{}, {}, {"_axis_0"}]>]
  //
  // Verify block args are converted (complex -> x2xf32):
  // CHECK-SAME: (%{{.*}}: tensor<16x16x2xf32>
  // CHECK-SAME:  %{{.*}}: tensor<32x128xbf16>
  // CHECK-SAME:  %{{.*}}: tensor<1x16x128xbf16>
  //
  // Verify no complex types remain inside the region:
  // CHECK-NOT: complex<f32>
  // CHECK: sdy.return
  %0 = sdy.manual_computation(%arg0, %arg1, %arg2)
    in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{"_axis_0"}, {}]>, <@mesh, [{}, {}, {}]>]
    out_shardings=[<@mesh, [{}, {}, {"_axis_0"}]>]
    manual_axes={"_axis_0_aux", "_axis_0"}
    (%a0: tensor<16x16xcomplex<f32>>, %a1: tensor<32x128xbf16>, %a2: tensor<1x16x128xbf16>) {
      // Linear projection (non-complex path)
      %1 = stablehlo.reshape %a2 : (tensor<1x16x128xbf16>) -> tensor<16x128xbf16>
      %2 = stablehlo.transpose %a1, dims = [1, 0] : (tensor<32x128xbf16>) -> tensor<128x32xbf16>
      %3 = stablehlo.dot_general %1, %2, contracting_dims = [1] x [0] : (tensor<16x128xbf16>, tensor<128x32xbf16>) -> tensor<16x32xbf16>
      %4 = stablehlo.reshape %3 : (tensor<16x32xbf16>) -> tensor<1x16x1x32xbf16>
      %5 = stablehlo.convert %4 : (tensor<1x16x1x32xbf16>) -> tensor<1x16x1x32xf32>
      %6 = stablehlo.reshape %5 : (tensor<1x16x1x32xf32>) -> tensor<1x16x1x16x2xf32>

      // Split into real/imag and construct complex
      %7 = stablehlo.slice %6 [0:1, 0:16, 0:1, 0:16, 0:1] : (tensor<1x16x1x16x2xf32>) -> tensor<1x16x1x16x1xf32>
      %8 = stablehlo.reshape %7 : (tensor<1x16x1x16x1xf32>) -> tensor<1x16x1x16xf32>
      %9 = stablehlo.slice %6 [0:1, 0:16, 0:1, 0:16, 1:2] : (tensor<1x16x1x16x2xf32>) -> tensor<1x16x1x16x1xf32>
      %10 = stablehlo.reshape %9 : (tensor<1x16x1x16x1xf32>) -> tensor<1x16x1x16xf32>
      %11 = stablehlo.complex %8, %10 : tensor<1x16x1x16xcomplex<f32>>

      // Complex RoPE multiply: h_complex * freqs
      %12 = stablehlo.reshape %a0 : (tensor<16x16xcomplex<f32>>) -> tensor<1x16x16xcomplex<f32>>
      %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 3] : (tensor<1x16x16xcomplex<f32>>) -> tensor<1x16x1x16xcomplex<f32>>
      %14 = stablehlo.multiply %11, %13 : tensor<1x16x1x16xcomplex<f32>>

      // Extract real/imag and flatten back
      %15 = stablehlo.real %14 : (tensor<1x16x1x16xcomplex<f32>>) -> tensor<1x16x1x16xf32>
      %16 = stablehlo.reshape %15 : (tensor<1x16x1x16xf32>) -> tensor<1x16x1x16x1xf32>
      %17 = stablehlo.imag %14 : (tensor<1x16x1x16xcomplex<f32>>) -> tensor<1x16x1x16xf32>
      %18 = stablehlo.reshape %17 : (tensor<1x16x1x16xf32>) -> tensor<1x16x1x16x1xf32>
      %19 = stablehlo.concatenate %16, %18, dim = 4 : (tensor<1x16x1x16x1xf32>, tensor<1x16x1x16x1xf32>) -> tensor<1x16x1x16x2xf32>
      %20 = stablehlo.reshape %19 : (tensor<1x16x1x16x2xf32>) -> tensor<1x16x32xf32>
      %21 = stablehlo.convert %20 : (tensor<1x16x32xf32>) -> tensor<1x16x32xbf16>
      sdy.return %21 : tensor<1x16x32xbf16>
  } : (tensor<16x16xcomplex<f32>>, tensor<128x128xbf16>, tensor<1x16x128xbf16>) -> tensor<1x16x128xbf16>
  return %0 : tensor<1x16x128xbf16>
}

// Test that ManualComputationOp with complex types created INSIDE the region
// (not as arguments or results) is still converted. This is the case the
// reviewer flagged: stablehlo.complex is created from real-valued slices
// inside the region and consumed by real/imag, without any complex-typed
// arguments or results on the ManualComputationOp itself.

// CHECK-LABEL: func.func @test_complex_created_inside_manual_computation
// CHECK-SAME: %arg0: tensor<1x16x1x32xf32>
// CHECK-SAME: %arg1: tensor<1x16x16x2xf32>
func.func @test_complex_created_inside_manual_computation(
    %arg0: tensor<1x16x1x32xf32>,
    %arg1: tensor<1x16x16x2xf32>
) -> tensor<1x16x32xf32> {
  // CHECK: sdy.manual_computation
  // CHECK-SAME: (%{{.*}}: tensor<1x16x1x32xf32>
  // CHECK-SAME:  %{{.*}}: tensor<1x16x16x2xf32>
  //
  // Verify no complex types remain inside the region:
  // CHECK-NOT: complex<f32>
  //
  // Verify complex ops were decomposed (concatenate from complex->pair):
  // CHECK: stablehlo.concatenate
  // CHECK-NOT: complex<f32>
  // CHECK: sdy.return
  %0 = sdy.manual_computation(%arg0, %arg1)
    in_shardings=[<@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>]
    out_shardings=[<@mesh, [{}, {}, {}]>]
    manual_axes={"_axis_0_aux", "_axis_0"}
    (%a0: tensor<1x16x1x32xf32>, %a1: tensor<1x16x16x2xf32>) {
      // Reshape input to expose real/imag pairs
      %1 = stablehlo.reshape %a0 : (tensor<1x16x1x32xf32>) -> tensor<1x16x1x16x2xf32>

      // Split into real and imag components
      %2 = stablehlo.slice %1 [0:1, 0:16, 0:1, 0:16, 0:1] : (tensor<1x16x1x16x2xf32>) -> tensor<1x16x1x16x1xf32>
      %3 = stablehlo.reshape %2 : (tensor<1x16x1x16x1xf32>) -> tensor<1x16x1x16xf32>
      %4 = stablehlo.slice %1 [0:1, 0:16, 0:1, 0:16, 1:2] : (tensor<1x16x1x16x2xf32>) -> tensor<1x16x1x16x1xf32>
      %5 = stablehlo.reshape %4 : (tensor<1x16x1x16x1xf32>) -> tensor<1x16x1x16xf32>

      // Create complex type INSIDE the region (this is the key scenario)
      %6 = stablehlo.complex %3, %5 : tensor<1x16x1x16xcomplex<f32>>

      // Use freqs (already real-valued, reshape to broadcast)
      %7 = stablehlo.reshape %a1 : (tensor<1x16x16x2xf32>) -> tensor<1x16x1x16x2xf32>
      %8 = stablehlo.slice %7 [0:1, 0:16, 0:1, 0:16, 0:1] : (tensor<1x16x1x16x2xf32>) -> tensor<1x16x1x16x1xf32>
      %9 = stablehlo.reshape %8 : (tensor<1x16x1x16x1xf32>) -> tensor<1x16x1x16xf32>
      %10 = stablehlo.slice %7 [0:1, 0:16, 0:1, 0:16, 1:2] : (tensor<1x16x1x16x2xf32>) -> tensor<1x16x1x16x1xf32>
      %11 = stablehlo.reshape %10 : (tensor<1x16x1x16x1xf32>) -> tensor<1x16x1x16xf32>
      %12 = stablehlo.complex %9, %11 : tensor<1x16x1x16xcomplex<f32>>

      // Complex multiply inside the region
      %13 = stablehlo.multiply %6, %12 : tensor<1x16x1x16xcomplex<f32>>

      // Extract real/imag and flatten back to real-valued output
      %14 = stablehlo.real %13 : (tensor<1x16x1x16xcomplex<f32>>) -> tensor<1x16x1x16xf32>
      %15 = stablehlo.reshape %14 : (tensor<1x16x1x16xf32>) -> tensor<1x16x1x16x1xf32>
      %16 = stablehlo.imag %13 : (tensor<1x16x1x16xcomplex<f32>>) -> tensor<1x16x1x16xf32>
      %17 = stablehlo.reshape %16 : (tensor<1x16x1x16xf32>) -> tensor<1x16x1x16x1xf32>
      %18 = stablehlo.concatenate %15, %17, dim = 4 : (tensor<1x16x1x16x1xf32>, tensor<1x16x1x16x1xf32>) -> tensor<1x16x1x16x2xf32>
      %19 = stablehlo.reshape %18 : (tensor<1x16x1x16x2xf32>) -> tensor<1x16x32xf32>
      sdy.return %19 : tensor<1x16x32xf32>
  } : (tensor<1x16x1x32xf32>, tensor<1x16x16x2xf32>) -> tensor<1x16x32xf32>
  return %0 : tensor<1x16x32xf32>
}
