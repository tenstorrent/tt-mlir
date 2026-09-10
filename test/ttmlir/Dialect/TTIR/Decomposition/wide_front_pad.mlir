// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

// tt-metal runs front padding on its row-major pad kernel, whose circular
// buffers hold 18 padded rows however many rows the tensor has. A 51275-wide
// bf16 row (100 KB) overflows L1, so the pad becomes ttir.full + ttir.concat.
// Narrow front pads and back-only pads keep ttir.pad.
module attributes {} {
  func.func @wide_front_pad(%arg0: tensor<1x1985xbf16>) -> tensor<1x51275xbf16> {
    // CHECK-LABEL: @wide_front_pad
    // CHECK: [[FILL:%.+]] = "ttir.full"()
    // CHECK-SAME: shape = array<i32: 1, 49290>
    // CHECK-SAME: -> tensor<1x49290xbf16>
    // CHECK: [[CAT:%.+]] = "ttir.concat"([[FILL]], %arg0)
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: -> tensor<1x51275xbf16>
    // CHECK-NOT: "ttir.pad"
    // CHECK: return [[CAT]]
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 49290, 0>, value = 1.0 : f32}> : (tensor<1x1985xbf16>) -> tensor<1x51275xbf16>
    return %0 : tensor<1x51275xbf16>
  }

  func.func @wide_front_and_back_pad_two_dims(%arg0: tensor<2x1985xbf16>) -> tensor<3x51275xbf16> {
    // CHECK-LABEL: @wide_front_and_back_pad_two_dims
    // CHECK: [[F0:%.+]] = "ttir.full"()
    // CHECK-SAME: shape = array<i32: 1, 1985>
    // CHECK: [[C0:%.+]] = "ttir.concat"([[F0]], %arg0)
    // CHECK-SAME: dim = 0 : si32
    // CHECK-SAME: -> tensor<3x1985xbf16>
    // CHECK: [[F1:%.+]] = "ttir.full"()
    // CHECK-SAME: shape = array<i32: 3, 49000>
    // CHECK: [[F2:%.+]] = "ttir.full"()
    // CHECK-SAME: shape = array<i32: 3, 290>
    // CHECK: [[C1:%.+]] = "ttir.concat"([[F1]], [[C0]], [[F2]])
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: -> tensor<3x51275xbf16>
    // CHECK-NOT: "ttir.pad"
    // CHECK: return [[C1]]
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 1, 0, 49000, 290>, value = 0.0 : f32}> : (tensor<2x1985xbf16>) -> tensor<3x51275xbf16>
    return %0 : tensor<3x51275xbf16>
  }

  func.func @narrow_front_pad_keeps_pad(%arg0: tensor<1x1985xbf16>) -> tensor<1x13115xbf16> {
    // CHECK-LABEL: @narrow_front_pad_keeps_pad
    // CHECK: "ttir.pad"
    // CHECK-SAME: padding = array<i32: 0, 0, 11130, 0>
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 11130, 0>, value = 1.0 : f32}> : (tensor<1x1985xbf16>) -> tensor<1x13115xbf16>
    return %0 : tensor<1x13115xbf16>
  }

  func.func @wide_back_pad_keeps_pad(%arg0: tensor<1x1985xbf16>) -> tensor<1x51275xbf16> {
    // CHECK-LABEL: @wide_back_pad_keeps_pad
    // CHECK: "ttir.pad"
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 49290>
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 49290>, value = 0.0 : f32}> : (tensor<1x1985xbf16>) -> tensor<1x51275xbf16>
    return %0 : tensor<1x51275xbf16>
  }
}
