// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// zeros_buffer serializes to the same flatbuffer op as ttnn.zeros
// (NamedFullOp with type = Zeros), so it executes as ttnn::zeros on device.
// const-eval is left at its default (enabled) so this also covers the
// no-hoisting requirement end to end.

module {
  func.func @zeros_buffer_2d() -> tensor<32x128xbf16> {
    // CHECK: {{.*}} = "ttnn.zeros_buffer"({{.*}})
    %0 = "ttir.zeros_buffer"() <{shape = array<i32: 32, 128>}> : () -> tensor<32x128xbf16>
    return %0 : tensor<32x128xbf16>
  }

  func.func @zeros_buffer_4d() -> tensor<8x16x32x128xbf16> {
    // CHECK: {{.*}} = "ttnn.zeros_buffer"({{.*}}) {{.*}} -> tensor<8x16x32x128xbf16{{.*}}>
    %0 = "ttir.zeros_buffer"() <{shape = array<i32: 8, 16, 32, 128>}> : () -> tensor<8x16x32x128xbf16>
    return %0 : tensor<8x16x32x128xbf16>
  }

  func.func @zeros_buffer_f32() -> tensor<32x64x128xf32> {
    // CHECK: {{.*}} = "ttnn.zeros_buffer"({{.*}}) {{.*}} -> tensor<32x64x128xf32{{.*}}>
    %0 = "ttir.zeros_buffer"() <{shape = array<i32: 32, 64, 128>}> : () -> tensor<32x64x128xf32>
    return %0 : tensor<32x64x128xf32>
  }

  // The reason the op exists: two identically-shaped caches, two allocations.
  func.func @kv_pair() -> (tensor<8x16x32x128xbf16>, tensor<8x16x32x128xbf16>) {
    // CHECK-LABEL: func.func @kv_pair
    // CHECK-NOT: ttcore.load_cached
    // CHECK: %[[K:.*]] = "ttnn.zeros_buffer"
    // CHECK: %[[V:.*]] = "ttnn.zeros_buffer"
    // CHECK: return %[[K]], %[[V]]
    %k = "ttir.zeros_buffer"() <{shape = array<i32: 8, 16, 32, 128>}> : () -> tensor<8x16x32x128xbf16>
    %v = "ttir.zeros_buffer"() <{shape = array<i32: 8, 16, 32, 128>}> : () -> tensor<8x16x32x128xbf16>
    return %k, %v : tensor<8x16x32x128xbf16>, tensor<8x16x32x128xbf16>
  }
}
