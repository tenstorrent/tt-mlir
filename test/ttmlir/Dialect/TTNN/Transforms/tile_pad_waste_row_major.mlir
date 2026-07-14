// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --mlir-print-local-scope %s | FileCheck %s

// A movement op stays row-major when tiling its input wastes >= 1 GiB (the
// HiDream RoPE freqs `...x2x2` blow-up). Shapes mimic the model.
module {
  // freqs slice: input 1x4480x1x64x2x2 tiles to ~1.17 GiB of padding -> row-major.
  // CHECK-LABEL: @rope_freqs_slice_row_major
  func.func @rope_freqs_slice_row_major(%arg0: tensor<1x4480x1x64x2x2xf32> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<1x4480x1x64x2x1xf32> {
    // CHECK: ttir.slice_static
    // CHECK-SAME: -> tensor<1x4480x1x64x2x1xf32, {{.*}}memref<{{[0-9x]+}}xf32,
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4480 : i32, 1 : i32, 64 : i32, 2 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4480x1x64x2x2xf32>) -> tensor<1x4480x1x64x2x1xf32>
    return %0 : tensor<1x4480x1x64x2x1xf32>
  }

  // cos/sin reshape: input 1x4480x1x64x1x1 (~1.17 GiB) collapsed in row-major.
  // CHECK-LABEL: @rope_cos_reshape_row_major
  func.func @rope_cos_reshape_row_major(%arg0: tensor<1x4480x1x64x1x1xf32> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<1x4480x1x64xf32> {
    // CHECK: ttir.reshape
    // CHECK-SAME: -> tensor<1x4480x1x64xf32, {{.*}}memref<{{[0-9x]+}}xf32,
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 4480 : i32, 1 : i32, 64 : i32]}> : (tensor<1x4480x1x64x1x1xf32>) -> tensor<1x4480x1x64xf32>
    return %0 : tensor<1x4480x1x64xf32>
  }

  // Tiny trailing dims but tiny tensor: waste << 1 GiB, so it still tilizes.
  // CHECK-LABEL: @small_waste_tilizes
  func.func @small_waste_tilizes(%arg0: tensor<1x64x2x2xf32> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<1x64x2x1xf32> {
    // CHECK: ttir.slice_static
    // CHECK-SAME: -> tensor<1x64x2x1xf32, {{.*}}!ttcore.tile<32x32
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 64 : i32, 2 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x2x2xf32>) -> tensor<1x64x2x1xf32>
    return %0 : tensor<1x64x2x1xf32>
  }
}
