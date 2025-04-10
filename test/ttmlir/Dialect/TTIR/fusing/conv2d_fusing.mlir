// RUN: ttmlir-opt --split-input-file --ttir-fusing %s | FileCheck %s

// Fuse add into conv. We also check that all uses of add are updated
module {
  func.func @conv2d_simple(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"
    // CHECK-SAME: dilation = 1
    // CHECK-SAME: groups = 1
    // CHECK-SAME: padding = 0
    // CHECK-SAME: stride = 1
    // CHECK-SAME: tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>
    // CHECK-NEXT: return %[[CONV]]
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.add"(%1, %arg2, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %4: tensor<1x30x30x64xbf16>
  }
}

// -----

// Fuse add into conv. Added few more ops after bias add to check if all uses are updated
module {
  func.func @conv2d_simple(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[CONV]] = "ttir.conv2d"
    // CHECK-SAME: dilation = 1
    // CHECK-SAME: groups = 1
    // CHECK-SAME: padding = 0
    // CHECK-SAME: stride = 1
    // CHECK-SAME: tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK: %[[ADD:.*]] = "ttir.add"
    // CHECK-SAME: (%[[CONV]], %arg2
    // CHECK-NEXT: return %[[ADD]]
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %4 = "ttir.add"(%1, %arg2, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %5 = ttir.empty() : tensor<1x30x30x64xbf16>
    %6 = "ttir.add"(%4, %arg2, %5) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %6: tensor<1x30x30x64xbf16>
  }
}

// -----

// Test that we cannot fuse conv2d and bias because it would break dominance order
module {
  func.func @conv2d_simple(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"
    // CHECK-SAME: dilation = 1
    // CHECK-SAME: groups = 1
    // CHECK-SAME: padding = 0
    // CHECK-SAME: stride = 1
    // CHECK-SAME: tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK-NEXT: ttir.empty
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK-NEXT: ttir.empty
    %3 = ttir.empty() : tensor<1x1x1x64xbf16>
    // CHECK-NEXT: %[[ADD:.*]] = "ttir.add"
    %4 = "ttir.add"(%1, %3, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK-NEXT: return %[[ADD]]
    return %4: tensor<1x30x30x64xbf16>
  }
}
