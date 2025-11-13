// RUN: ttmlir-opt --ttnn-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test fusing sigmoid activation into matmul operation
module {
    func.func @matmul_sigmoid(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[MATMUL:[0-9]+]] = "ttnn.matmul"(%arg0, %arg1)
        // CHECK-SAME: activation = "sigmoid"
        // CHECK-NOT: ttnn.sigmoid
        // CHECK: return %[[MATMUL]]
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %1 : tensor<64x256xbf16>
    }
}

// Test matmul without activation (should not be modified)
module {
    func.func @matmul_no_activation(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[MATMUL:[0-9]+]] = "ttnn.matmul"(%arg0, %arg1)
        // CHECK-NOT: activation
        // CHECK: return %[[MATMUL]]
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
        return %0 : tensor<64x256xbf16>
    }
}

// Test matmul with multiple uses (should not fuse)
module {
    func.func @matmul_multiple_uses(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> (tensor<64x256xbf16>, tensor<64x256xbf16>) {
        // CHECK: %[[MATMUL:[0-9]+]] = "ttnn.matmul"(%arg0, %arg1)
        // CHECK-NOT: activation
        // CHECK: %[[SIGMOID:[0-9]+]] = "ttnn.sigmoid"(%[[MATMUL]])
        // CHECK: return %[[MATMUL]], %[[SIGMOID]]
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %0, %1 : tensor<64x256xbf16>, tensor<64x256xbf16>
    }
}

// Test matmul with activation already set (should not fuse)
module {
    func.func @matmul_activation_already_set(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[MATMUL:[0-9]+]] = "ttnn.matmul"(%arg0, %arg1)
        // CHECK-SAME: activation = "relu"
        // CHECK: %[[SIGMOID:[0-9]+]] = "ttnn.sigmoid"(%[[MATMUL]])
        // CHECK: return %[[SIGMOID]]
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false, activation = "relu"}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %1 : tensor<64x256xbf16>
    }
}

// Test matmul with transpose flags and sigmoid
module {
    func.func @matmul_transpose_sigmoid(%arg0: tensor<64x128xbf16>, %arg1: tensor<256x128xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[MATMUL:[0-9]+]] = "ttnn.matmul"(%arg0, %arg1)
        // CHECK-SAME: activation = "sigmoid"
        // CHECK-SAME: transpose_a = false
        // CHECK-SAME: transpose_b = true
        // CHECK-NOT: ttnn.sigmoid
        // CHECK: return %[[MATMUL]]
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<64x128xbf16>, tensor<256x128xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %1 : tensor<64x256xbf16>
    }
}

// Test matmul with different activation (relu, should not fuse)
module {
    func.func @matmul_relu_not_sigmoid(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x256xbf16>) -> tensor<64x256xbf16> {
        // CHECK: %[[MATMUL:[0-9]+]] = "ttnn.matmul"(%arg0, %arg1)
        // CHECK-NOT: activation
        // CHECK: %[[RELU:[0-9]+]] = "ttnn.relu"(%[[MATMUL]])
        // CHECK: return %[[RELU]]
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
        %1 = "ttnn.relu"(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
        return %1 : tensor<64x256xbf16>
    }
}

// Test batched matmul with sigmoid
module {
    func.func @batched_matmul_sigmoid(%arg0: tensor<8x64x128xbf16>, %arg1: tensor<8x128x256xbf16>) -> tensor<8x64x256xbf16> {
        // CHECK: %[[MATMUL:[0-9]+]] = "ttnn.matmul"(%arg0, %arg1)
        // CHECK-SAME: activation = "sigmoid"
        // CHECK-NOT: ttnn.sigmoid
        // CHECK: return %[[MATMUL]]
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<8x64x128xbf16>, tensor<8x128x256xbf16>) -> tensor<8x64x256xbf16>
        %1 = "ttnn.sigmoid"(%0) : (tensor<8x64x256xbf16>) -> tensor<8x64x256xbf16>
        return %1 : tensor<8x64x256xbf16>
    }
}
