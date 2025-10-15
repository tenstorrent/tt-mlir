// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test relu6 folding from clamp operation with scalar constants
func.func @clamp_scalar_relu6(%arg0: tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16> {
    %0 = ttir.empty() : tensor<8x32x112x112xbf16>
    // CHECK: %[[RELU6:.*]] = "ttir.relu6"
    // CHECK-NOT: "ttir.clamp_scalar"
    %1 = "ttir.clamp_scalar"(%arg0, %0) <{max = 6.000000e+00 : f32, min = 0.000000e+00 : f32}> : (tensor<8x32x112x112xbf16>, tensor<8x32x112x112xbf16>) -> tensor<8x32x112x112xbf16>
    // CHECK: return %[[RELU6]]
    return %1 : tensor<8x32x112x112xbf16>
}
// Test relu6 folding from two chained clamp operations
func.func @clamp_scalar_chained_relu(%arg0: tensor<8x1x1x256xbf16>) -> tensor<8x1x1x256xbf16> {
    %0 = ttir.empty() : tensor<8x1x1x256xbf16>
    %1 = ttir.empty() : tensor<8x1x1x256xbf16>
    // CHECK: %[[RELU6:.*]] = "ttir.relu6"
    // CHECK-NOT: "ttir.clamp_scalar"
    %2 = "ttir.clamp_scalar"(%arg0, %0) <{max = 0x7F800000 : f32, min = 0.000000e+00 : f32}> : (tensor<8x1x1x256xbf16>, tensor<8x1x1x256xbf16>) -> tensor<8x1x1x256xbf16>
    %3 = "ttir.clamp_scalar"(%2, %1) <{max = 6.000000e+00 : f32, min = 0xFF800000 : f32}> : (tensor<8x1x1x256xbf16>, tensor<8x1x1x256xbf16>) -> tensor<8x1x1x256xbf16>
    // CHECK: return %[[RELU6]]
    return %3 : tensor<8x1x1x256xbf16>
}