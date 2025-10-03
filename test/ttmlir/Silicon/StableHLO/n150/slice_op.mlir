// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s -o %t.mlir --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%"
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @mod_slice attributes {} {
  func.func public @test_slice(%arg0: tensor<32x64xbf16>) -> tensor<8x8xbf16> {
    // CHECK-LABEL: func.func public @test_slice
    // CHECK: ttnn.slice_static
    // CHECK-SAME: begins = [0 : i32, 16 : i32],
    // CHECK-SAME: ends = [16 : i32, 32 : i32],
    // CHECK-SAME: step = [2 : i32, 2 : i32]
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: -> tensor<8x8xbf16
    %result = "stablehlo.slice"(%arg0) {
      start_indices = array<i64: 0, 16>,
      limit_indices = array<i64: 16, 32>,
      strides = array<i64: 2, 2>
    } : (tensor<32x64xbf16>) -> tensor<8x8xbf16>
    return %result : tensor<8x8xbf16>
  }

  func.func public @test_slice_f32(%arg0: tensor<32x64xf32>) -> (tensor<16x16xf32>) {
    // CHECK-LABEL: @test_slice_f32(
    // CHECK: ttnn.slice_static
    // CHECK-SAME: begins = [0 : i32, 16 : i32],
    // CHECK-SAME: ends = [16 : i32, 32 : i32],
    // CHECK-SAME: step = [1 : i32, 1 : i32]
    // CHECK-SAME: tensor<32x64xf32
    // CHECK-SAME: -> tensor<16x16xf32
    %0 = stablehlo.slice %arg0 [0:16, 16:32] : (tensor<32x64xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func public @test_slice_non_tilize(%arg0: tensor<32x64xf32>) -> (tensor<14x14xf32>) {
    // CHECK-LABEL: @test_slice_non_tilize(
    // CHECK: ttnn.slice_static
    // CHECK-SAME: begins = [0 : i32, 16 : i32],
    // CHECK-SAME: ends = [14 : i32, 30 : i32],
    // CHECK-SAME: step = [1 : i32, 1 : i32]
    // CHECK-SAME: tensor<32x64xf32
    // CHECK-SAME: -> tensor<14x14xf32
    %0 = stablehlo.slice %arg0 [0:14, 16:30] : (tensor<32x64xf32>) -> tensor<14x14xf32>
    return %0 : tensor<14x14xf32>
  }
}
