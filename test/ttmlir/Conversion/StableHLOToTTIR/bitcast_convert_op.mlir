// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_bitcast_convert attributes {} {
  func.func public @test_bitcast_convert_ui32_to_f32(%arg0: tensor<64xui32>) -> tensor<64xf32> {
    %0 = stablehlo.bitcast_convert %arg0 : (tensor<64xui32>) -> tensor<64xf32>
    // CHECK: = "ttir.bitcast_convert"
    // CHECK-SAME: (tensor<64xui32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func public @test_bitcast_convert_f32_to_ui32(%arg0: tensor<64xf32>) -> tensor<64xui32> {
    %0 = stablehlo.bitcast_convert %arg0 : (tensor<64xf32>) -> tensor<64xui32>
    // CHECK: = "ttir.bitcast_convert"
    // CHECK-SAME: (tensor<64xf32>) -> tensor<64xui32>
    return %0 : tensor<64xui32>
  }
  func.func public @test_bitcast_convert_bf16_to_ui16(%arg0: tensor<32x32xbf16>) -> tensor<32x32xui16> {
    %0 = stablehlo.bitcast_convert %arg0 : (tensor<32x32xbf16>) -> tensor<32x32xui16>
    // CHECK: = "ttir.bitcast_convert"
    // CHECK-SAME: (tensor<32x32xbf16>) -> tensor<32x32xui16>
    return %0 : tensor<32x32xui16>
  }
  func.func public @test_bitcast_convert_more_dims(%arg0: tensor<5x32x32x3xbf16>) -> tensor<5x32x32x3xui16> {
    %0 = stablehlo.bitcast_convert %arg0 : (tensor<5x32x32x3xbf16>) -> tensor<5x32x32x3xui16>
    // CHECK: = "ttir.bitcast_convert"
    // CHECK-SAME: (tensor<5x32x32x3xbf16>) -> tensor<5x32x32x3xui16>
    return %0 : tensor<5x32x32x3xui16>
  }
}
