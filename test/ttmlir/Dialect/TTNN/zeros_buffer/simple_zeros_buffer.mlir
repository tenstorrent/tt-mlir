// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Note: const-eval is left at its default (enabled) on purpose. The whole point
// of zeros_buffer is that N calls survive as N distinct allocations under
// the default pipeline, so a test that disabled const-eval would prove nothing.

module {
  // Two identically-shaped caches must remain two separate ops: neither merged
  // by CSE nor hoisted behind ttcore.load_cached.
  func.func @two_caches_stay_distinct() -> (tensor<64x128xbf16>, tensor<64x128xbf16>) {
    // CHECK-LABEL: func.func @two_caches_stay_distinct
    // CHECK-NOT: ttcore.load_cached
    // CHECK: %[[K:.*]] = "ttnn.zeros_buffer"
    // CHECK-SAME: shape = #ttnn.shape<64x128>
    // CHECK: %[[V:.*]] = "ttnn.zeros_buffer"
    // CHECK-SAME: shape = #ttnn.shape<64x128>
    // CHECK-NOT: ttcore.load_cached
    // CHECK: return %[[K]], %[[V]]
    %k = "ttir.zeros_buffer"() <{shape = array<i32: 64, 128>}> : () -> tensor<64x128xbf16>
    %v = "ttir.zeros_buffer"() <{shape = array<i32: 64, 128>}> : () -> tensor<64x128xbf16>
    return %k, %v : tensor<64x128xbf16>, tensor<64x128xbf16>
  }

  // No const-eval function may be created for this op.
  // CHECK-NOT: _const_eval_

  func.func @zeros_buffer_4d() -> tensor<13x24x56x42xbf16> {
    // CHECK-LABEL: func.func @zeros_buffer_4d
    // CHECK: "ttnn.zeros_buffer"
    // CHECK-SAME: -> tensor<13x24x56x42xbf16{{.*}}>
    %0 = "ttir.zeros_buffer"() <{shape = array<i32: 13, 24, 56, 42>}> : () -> tensor<13x24x56x42xbf16>
    return %0 : tensor<13x24x56x42xbf16>
  }

  func.func @zeros_buffer_f32() -> tensor<32x64x128xf32> {
    // CHECK-LABEL: func.func @zeros_buffer_f32
    // CHECK: "ttnn.zeros_buffer"
    // CHECK-SAME: -> tensor<32x64x128xf32{{.*}}>
    %0 = "ttir.zeros_buffer"() <{shape = array<i32: 32, 64, 128>}> : () -> tensor<32x64x128xf32>
    return %0 : tensor<32x64x128xf32>
  }
}
