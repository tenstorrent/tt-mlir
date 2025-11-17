// RUN: ttmlir-opt --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// Fuse add into conv. We also check that all uses of add are updated.
module {
  // CHECK-LABEL: func.func @conv2d_fuse
  func.func @conv2d_fuse(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: dilation = 1
    // CHECK-SAME: groups = 1
    // CHECK-SAME: padding = 0
    // CHECK-SAME: stride = 1
    // CHECK-NOT: "ttir.add"
    // CHECK-NEXT: return %[[CONV]]
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.add"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %4: tensor<1x30x30x64xbf16>
  }
}

// Test that we can fuse conv2d and bias even when bias is defined after conv2d because we reorder ops.
module {
  // CHECK-LABEL: func.func @conv2d_dominance_order
  func.func @conv2d_dominance_order(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"
    // CHECK-SAME: dilation = 1
    // CHECK-SAME: groups = 1
    // CHECK-SAME: padding = 0
    // CHECK-SAME: stride = 1
    // CHECK-SAME: tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>
    // CHECK-NOT: "ttir.add"
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %3 = ttir.empty() : tensor<1x1x1x64xbf16>
    %4 = "ttir.add"(%1, %3, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %4: tensor<1x30x30x64xbf16>
  }
}

// Test that we cannot fuse conv2d and bias because conv2d has more than one use.
module {
  // CHECK-LABEL: func.func @conv2d_multiple_uses
  func.func @conv2d_multiple_uses(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"(%arg0, %arg1, %0)
    // CHECK-SAME: dilation = 1
    // CHECK-SAME: groups = 1
    // CHECK-SAME: padding = 0
    // CHECK-SAME: stride = 1
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK: ttir.add
    // CHECK: ttir.add
    // CHECK: %[[RETURNADD:.*]] = "ttir.add"
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    // First use of conv2d.
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %4 = ttir.empty() : tensor<1x30x30x64xbf16>
    // Second use of conv2d (we cannot fuse).
    %5 = "ttir.add"(%1, %arg2, %4) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %6 = ttir.empty() : tensor<1x30x30x64xbf16>
    %7 = "ttir.add"(%3, %5, %6) : (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK-NEXT: return %[[RETURNADD]]
    return %7: tensor<1x30x30x64xbf16>
  }
}

// Check that we can fuse two adds into conv2d bias.
module {
  // CHECK-LABEL: func.func @conv2d_double_add
  func.func @conv2d_double_add(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[EMPTY:.*]] = ttir.empty() : tensor<1x1x1x64xbf16>
    // CHECK-NEXT: %[[BIASADD:.*]] = "ttir.add"(%arg2, %arg2, %[[EMPTY]])
    // CHECK-NEXT: %[[CONV:.*]] = "ttir.conv2d"(%arg0, %arg1, %[[BIASADD]], %0)
    // CHECK-SAME: dilation = 1
    // CHECK-SAME: groups = 1
    // CHECK-SAME: padding = 0
    // CHECK-SAME: stride = 1
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
    <{
    stride = 1: i32,
    padding = 0: i32,
    dilation = 1: i32,
    groups = 1: i32
    }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    // First add is fused into bias.
    %4 = "ttir.add"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %5 = ttir.empty() : tensor<1x30x30x64xbf16>
    // Second add is also fused into bias (arg2 + arg2).
    %6 = "ttir.add"(%4, %arg2, %5) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK-NEXT: return %[[CONV]]
    return %6: tensor<1x30x30x64xbf16>
  }
}

// Check that we cannot fuse add because second argument to add next to conv is not suitable for bias.
module {
  // CHECK-LABEL: func.func @conv2d_not_suitable_for_bias
  func.func @conv2d_not_suitable_for_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // We will add this empty tensor to conv2d output. This cannot be fused this its not in the right format.
    %1 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"(%arg0, %arg1, %0)
    // CHECK-SAME: dilation = 1
    // CHECK-SAME: groups = 1
    // CHECK-SAME: padding = 0
    // CHECK-SAME: stride = 1
    %2 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK: %[[RETURNADD:.*]] = "ttir.add"(%[[CONV]]
    %3 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.add"(%2, %1, %3) : (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK-NEXT: return %[[RETURNADD]]
    return %4: tensor<1x30x30x64xbf16>
  }
}

// Test that we can fuse nontrivial bias patterns into conv2d.
module {
  // CHECK-LABEL: func.func @conv2d_nontrivial_bias
  func.func @conv2d_nontrivial_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>, %arg3: tensor<1x1x1x64xbf16>, %arg4: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"
    // CHECK-SAME: dilation = 1
    // CHECK-SAME: groups = 1
    // CHECK-SAME: padding = 0
    // CHECK-SAME: stride = 1
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    %2 = ttir.empty() : tensor<1x1x1x64xbf16>
    %3 = "ttir.add"(%arg2, %arg3, %2) : (tensor<1x1x1x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xbf16>
    %4 = ttir.empty() : tensor<1x1x1x64xbf16>
    %5 = "ttir.multiply"(%3, %arg4, %4) : (tensor<1x1x1x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xbf16>

    %6 = ttir.empty() : tensor<1x30x30x64xbf16>
    %7 = "ttir.add"(%1, %5, %6) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    // CHECK-NEXT: return %[[CONV]]
    return %7: tensor<1x30x30x64xbf16>
  }
}
