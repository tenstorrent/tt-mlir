// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @main(%arg0: tensor<1x128x128x384xf32>) -> tensor<1x132x132x384xf32> {
    // CHECK: ttir.pad
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 4, 0, 4, 0, 0>
    // CHECK-SAME: value = 0.000000e+00 : f32
    // CHECK-SAME: (tensor<1x128x128x384xf32>) -> tensor<1x132x132x384xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1xf64>
    %0 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xf32>
    %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
    %2 = stablehlo.pad %arg0, %1, low = [0, 0, 0, 0], high = [0, 4, 4, 0], interior = [0, 0, 0, 0] : (tensor<1x128x128x384xf32>, tensor<f32>) -> tensor<1x132x132x384xf32>
    return %2 : tensor<1x132x132x384xf32>
  }
}

module {
  func.func @main(%arg0: tensor<1x64x64x384xf32>) -> tensor<1x72x72x384xf32> {
    // CHECK: ttir.pad
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 8, 0, 8, 0, 0>
    // CHECK-SAME: value = 0.000000e+00 : f32
    // CHECK-SAME: (tensor<1x64x64x384xf32>) -> tensor<1x72x72x384xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1xf64>
    %0 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xf32>
    %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
    %2 = stablehlo.pad %arg0, %1, low = [0, 0, 0, 0], high = [0, 8, 8, 0], interior = [0, 0, 0, 0] : (tensor<1x64x64x384xf32>, tensor<f32>) -> tensor<1x72x72x384xf32>
    return %2 : tensor<1x72x72x384xf32>
  }
}

module {
  func.func @main(%arg0: tensor<1x64x64x768xbf16>) -> tensor<1x72x72x768xbf16> {
    // CHECK: ttir.pad
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 8, 0, 8, 0, 0>
    // CHECK-SAME: value = 0.000000e+00 : f32
    // CHECK-SAME: (tensor<1x64x64x768xbf16>) -> tensor<1x72x72x768xbf16>
    %cst = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %0 = stablehlo.convert %cst : (tensor<1xbf16>) -> tensor<1xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1xbf16>) -> tensor<bf16>
    %2 = stablehlo.pad %arg0, %1, low = [0, 0, 0, 0], high = [0, 8, 8, 0], interior = [0, 0, 0, 0] : (tensor<1x64x64x768xbf16>, tensor<bf16>) -> tensor<1x72x72x768xbf16>
    return %2 : tensor<1x72x72x768xbf16>
  }
}

module {
  func.func @main(%arg0: tensor<1x64x64x768xi32>) -> tensor<1x72x72x768xi32> {
    // CHECK: ttir.pad
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 8, 0, 8, 0, 0>
    // CHECK-SAME: value = 0.000000e+00 : f32
    // CHECK-SAME: (tensor<1x64x64x768xi32>) -> tensor<1x72x72x768xi32>
    %cst = arith.constant dense<0> : tensor<1xi32>
    %0 = stablehlo.convert %cst : (tensor<1xi32>) -> tensor<1xi32>
    %1 = stablehlo.reshape %0 : (tensor<1xi32>) -> tensor<i32>
    %2 = stablehlo.pad %arg0, %1, low = [0, 0, 0, 0], high = [0, 8, 8, 0], interior = [0, 0, 0, 0] : (tensor<1x64x64x768xi32>, tensor<i32>) -> tensor<1x72x72x768xi32>
    return %2 : tensor<1x72x72x768xi32>
  }
}

module {
  func.func public @main(%arg0: tensor<4x54x54x64xbf16> ) -> (tensor<4x54x54x68xbf16>) {
    // CHECK: ttir.pad
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 2, 2>
    // CHECK-SAME: value = 0.000000e+00 : f32
    // CHECK-SAME: (tensor<4x54x54x64xbf16>) -> tensor<4x54x54x68xbf16>
    %cst = stablehlo.constant dense<0> : tensor<i32>
    %0 = call @_pad(%arg0, %cst) : (tensor<4x54x54x64xbf16>, tensor<i32>) -> tensor<4x54x54x68xbf16>
    return %0 : tensor<4x54x54x68xbf16>
  }

  func.func private @_pad(%arg0: tensor<4x54x54x64xbf16>, %arg1: tensor<i32>) -> tensor<4x54x54x68xbf16> {
    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<bf16>
    %1 = stablehlo.pad %arg0, %0, low = [0, 0, 0, 2], high = [0, 0, 0, 2], interior = [0, 0, 0, 0] : (tensor<4x54x54x64xbf16>, tensor<bf16>) -> tensor<4x54x54x68xbf16>
    return %1 : tensor<4x54x54x68xbf16>
  }
}

// Interior padding only: dilate dims 1 and 2 by inserting 1 padding value
// between consecutive elements. Lowered via reshape/full/concat/reshape/slice
// (one dilation pass per padded dimension), no edge ttir.pad expected.
module {
  func.func @main(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x7x7x8xf32> {
    // CHECK: ttir.reshape
    // CHECK-SAME: -> tensor<1x4x1x4x8xf32>
    // CHECK: ttir.full
    // CHECK-SAME: value = 0.000000e+00 : f32
    // CHECK-SAME: -> tensor<1x4x1x4x8xf32>
    // CHECK: ttir.concat
    // CHECK-SAME: -> tensor<1x4x2x4x8xf32>
    // CHECK: ttir.reshape
    // CHECK-SAME: -> tensor<1x8x4x8xf32>
    // CHECK: ttir.slice_static
    // CHECK-SAME: -> tensor<1x7x4x8xf32>
    // CHECK: ttir.reshape
    // CHECK-SAME: -> tensor<1x7x4x1x8xf32>
    // CHECK: ttir.full
    // CHECK-SAME: -> tensor<1x7x4x1x8xf32>
    // CHECK: ttir.concat
    // CHECK-SAME: -> tensor<1x7x4x2x8xf32>
    // CHECK: ttir.reshape
    // CHECK-SAME: -> tensor<1x7x8x8xf32>
    // CHECK: ttir.slice_static
    // CHECK-SAME: -> tensor<1x7x7x8xf32>
    // CHECK-NOT: ttir.pad
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 1, 1, 0] : (tensor<1x4x4x8xf32>, tensor<f32>) -> tensor<1x7x7x8xf32>
    return %0 : tensor<1x7x7x8xf32>
  }
}

// Interior padding combined with edge padding on the same dimension: interior
// dilation runs first, then a single ttir.pad applies the low/high edges.
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<2x9xf32> {
    // CHECK: ttir.reshape
    // CHECK-SAME: -> tensor<1x4x1xf32>
    // CHECK: ttir.full
    // CHECK-SAME: -> tensor<1x4x1xf32>
    // CHECK: ttir.concat
    // CHECK-SAME: -> tensor<1x4x2xf32>
    // CHECK: ttir.reshape
    // CHECK-SAME: -> tensor<1x8xf32>
    // CHECK: ttir.slice_static
    // CHECK-SAME: -> tensor<1x7xf32>
    // CHECK: ttir.pad
    // CHECK-SAME: padding = array<i32: 1, 0, 1, 1>
    // CHECK-SAME: value = 0.000000e+00 : f32
    // CHECK-SAME: -> tensor<2x9xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.pad %arg0, %cst, low = [1, 1], high = [0, 1], interior = [0, 1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<2x9xf32>
    return %0 : tensor<2x9xf32>
  }
}

// Interior padding on an integer tensor: the fill value is emitted as an f32
// attribute on ttir.full (converted to the tensor element type downstream).
module {
  func.func @main(%arg0: tensor<1x3x3x4xi32>) -> tensor<1x5x7x4xi32> {
    // CHECK: ttir.full
    // CHECK-SAME: value = 0.000000e+00 : f32
    // CHECK-SAME: -> tensor<1x3x1x3x4xi32>
    // CHECK: ttir.concat
    // CHECK-SAME: -> tensor<1x3x2x3x4xi32>
    // CHECK: ttir.slice_static
    // CHECK-SAME: -> tensor<1x5x3x4xi32>
    // CHECK: ttir.full
    // CHECK-SAME: -> tensor<1x5x3x2x4xi32>
    // CHECK: ttir.concat
    // CHECK-SAME: -> tensor<1x5x3x3x4xi32>
    // CHECK: ttir.slice_static
    // CHECK-SAME: -> tensor<1x5x7x4xi32>
    // CHECK-NOT: ttir.pad
    %cst = arith.constant dense<0> : tensor<i32>
    %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 1, 2, 0] : (tensor<1x3x3x4xi32>, tensor<i32>) -> tensor<1x5x7x4xi32>
    return %0 : tensor<1x5x7x4xi32>
  }
}
