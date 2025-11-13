// RUN: ttmlir-opt --ttnn-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test fusing sigmoid activation into linear operation
module {
    func.func @linear_sigmoid(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[LINEAR:[0-9]+]] = "ttnn.linear"(%arg0, %arg1)
        // CHECK-SAME: activation = "sigmoid"
        // CHECK-NOT: ttnn.sigmoid
        // CHECK: return %[[LINEAR]]
        %0 = "ttnn.linear"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %1 : tensor<64x256xbf16>
    }
}

// Test linear with bias and sigmoid activation
module {
    func.func @linear_bias_sigmoid(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>, %arg2: tensor<256xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[LINEAR:[0-9]+]] = "ttnn.linear"(%arg0, %arg1, %arg2)
        // CHECK-SAME: activation = "sigmoid"
        // CHECK-NOT: ttnn.sigmoid
        // CHECK: return %[[LINEAR]]
        %0 = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x256xbf16>, tensor<256xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %1 : tensor<64x256xbf16>
    }
}

// Test linear without activation (should not be modified)
module {
    func.func @linear_no_activation(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[LINEAR:[0-9]+]] = "ttnn.linear"(%arg0, %arg1)
        // CHECK-NOT: activation
        // CHECK: return %[[LINEAR]]
        %0 = "ttnn.linear"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
        return %0 : tensor<64x256xbf16>
    }
}

// Test linear with multiple uses (should not fuse)
module {
    func.func @linear_multiple_uses(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> (tensor<64x256xbf16>, tensor<64x256xbf16>) {
        // CHECK: %[[LINEAR:[0-9]+]] = "ttnn.linear"(%arg0, %arg1)
        // CHECK-NOT: activation
        // CHECK: %[[SIGMOID:[0-9]+]] = "ttnn.sigmoid"(%[[LINEAR]])
        // CHECK: return %[[LINEAR]], %[[SIGMOID]]
        %0 = "ttnn.linear"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %0, %1 : tensor<64x256xbf16>, tensor<64x256xbf16>
    }
}

// Test linear with activation already set (should not fuse)
module {
    func.func @linear_activation_already_set(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[LINEAR:[0-9]+]] = "ttnn.linear"(%arg0, %arg1)
        // CHECK-SAME: activation = "gelu"
        // CHECK: %[[SIGMOID:[0-9]+]] = "ttnn.sigmoid"(%[[LINEAR]])
        // CHECK: return %[[SIGMOID]]
        %0 = "ttnn.linear"(%arg0, %arg1) <{transpose_a = false, transpose_b = false, activation = "gelu"}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %1 : tensor<64x256xbf16>
    }
}

// Test linear with different activation (relu, should not fuse)
module {
    func.func @linear_relu_not_sigmoid(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[LINEAR:[0-9]+]] = "ttnn.linear"(%arg0, %arg1)
        // CHECK-NOT: activation
        // CHECK: %[[RELU:[0-9]+]] = "ttnn.relu"(%[[LINEAR]])
        // CHECK: return %[[RELU]]
        %0 = "ttnn.linear"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.relu"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %1 : tensor<64x256xbf16>
    }
}

// Test linear with transpose flags and sigmoid
module {
    func.func @linear_transpose_sigmoid(%arg0: tensor<64x128xbf16>, %arg1: tensor<256x128xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[LINEAR:[0-9]+]] = "ttnn.linear"(%arg0, %arg1)
        // CHECK-SAME: activation = "sigmoid"
        // CHECK-SAME: transpose_a = false
        // CHECK-SAME: transpose_b = true
        // CHECK-NOT: ttnn.sigmoid
        // CHECK: return %[[LINEAR]]
        %0 = "ttnn.linear"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<64x128xbf16>, tensor<256x128xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %1 : tensor<64x256xbf16>
    }
}

// Test batched linear with sigmoid
module {
    func.func @batched_linear_sigmoid(%arg0: tensor<8x64x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<8x64x256xbf16> {
        // CHECK: %[[LINEAR:[0-9]+]] = "ttnn.linear"(%arg0, %arg1)
        // CHECK-SAME: activation = "sigmoid"
        // CHECK-NOT: ttnn.sigmoid
        // CHECK: return %[[LINEAR]]
        %0 = "ttnn.linear"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<8x64x128xbf16>, tensor<128x256xbf16>) -> tensor<8x64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<8x64x256xbf16>) -> tensor<8x64x256xbf16>
        return %1 : tensor<8x64x256xbf16>
    }
}
