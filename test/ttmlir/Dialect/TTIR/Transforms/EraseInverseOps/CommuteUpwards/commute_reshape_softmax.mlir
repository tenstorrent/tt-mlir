// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true enable-commute-downwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
    func.func @test_reshape_softmax_commute_flatten(%arg0: tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: shape = [1 : i32, 25600 : i32, 128 : i32]
        // CHECK: %[[SOFTMAX:[0-9]+]] = "ttir.softmax"(%[[RESHAPE]]
        // CHECK: dimension = 2
        // CHECK: return %[[SOFTMAX]]
        %1 = "ttir.softmax"(%arg0) <{dimension = 3 : si32}> : (tensor<1x160x160x128xbf16>) -> tensor<1x160x160x128xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16>
        return %2 : tensor<1x25600x128xbf16>
    }
    func.func @test_reshape_softmax_commute_unflatten(%arg0: tensor<1x25600x128xbf16>) -> tensor<1x160x160x128xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]
        // CHECK: %[[SOFTMAX:[0-9]+]] = "ttir.softmax"(%[[RESHAPE]]
        // CHECK: dimension = 3
        // CHECK: return %[[SOFTMAX]]
        %1 = "ttir.softmax"(%arg0) <{dimension = 2 : si32}> : (tensor<1x25600x128xbf16>) -> tensor<1x25600x128xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]}> : (tensor<1x25600x128xbf16>) -> tensor<1x160x160x128xbf16>
        return %2 : tensor<1x160x160x128xbf16>
    }
    func.func @test_reshape_softmax_no_commute_merge_softmax_dim(%arg0: tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16> {
        // CHECK: %[[SOFTMAX:[0-9]+]] = "ttir.softmax"(%arg0
        // CHECK: dimension = 1
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SOFTMAX]]
        // CHECK: shape = [1 : i32, 25600 : i32, 128 : i32]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.softmax"(%arg0) <{dimension = 1 : si32}> : (tensor<1x160x160x128xbf16>) -> tensor<1x160x160x128xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16>
        return %2 : tensor<1x25600x128xbf16>
    }
    func.func @test_reshape_softmax_no_commute_split_softmax_dim(%arg0: tensor<1x25600x128xbf16>) -> tensor<1x160x160x128xbf16> {
        // CHECK: %[[SOFTMAX:[0-9]+]] = "ttir.softmax"(%arg0
        // CHECK: dimension = 1
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SOFTMAX]]
        // CHECK: shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.softmax"(%arg0) <{dimension = 1 : si32}> : (tensor<1x25600x128xbf16>) -> tensor<1x25600x128xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]}> : (tensor<1x25600x128xbf16>) -> tensor<1x160x160x128xbf16>
        return %2 : tensor<1x160x160x128xbf16>
    }
    func.func @test_reshape_softmax_commute_negative_dim(%arg0: tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: shape = [1 : i32, 25600 : i32, 128 : i32]
        // CHECK: %[[SOFTMAX:[0-9]+]] = "ttir.softmax"(%[[RESHAPE]]
        // CHECK: dimension = 2
        // CHECK: return %[[SOFTMAX]]
        %1 = "ttir.softmax"(%arg0) <{dimension = -1 : si32}> : (tensor<1x160x160x128xbf16>) -> tensor<1x160x160x128xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16>
        return %2 : tensor<1x25600x128xbf16>
    }
    // If softmax has multiple users that are same reshapes, they can commute upwards.
    func.func @test_reshape_softmax_commute_two_same_reshapes(%arg0: tensor<1x160x160x128xbf16>) -> (tensor<1x25600x128xbf16>, tensor<1x25600x128xbf16>) {
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.softmax"
        %1 = "ttir.softmax"(%arg0) <{dimension = 3 : si32}> : (tensor<1x160x160x128xbf16>) -> tensor<1x160x160x128xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16>
        %3 = "ttir.reshape"(%1) <{shape = [1 : i32, 25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16>
        return %2, %3 : tensor<1x25600x128xbf16>, tensor<1x25600x128xbf16>
    }
    // If softmax has multiple and not all are same reshapes, they cannot commute upwards.
    func.func @test_reshape_softmax_commute_two_different_reshapes(%arg0: tensor<1x160x160x128xbf16>) -> (tensor<1x1x25600x128xbf16>, tensor<1x10x2560x128xbf16>) {
        // CHECK: "ttir.softmax"
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.reshape"
        %1 = "ttir.softmax"(%arg0) <{dimension = 3 : si32}> : (tensor<1x160x160x128xbf16>) -> tensor<1x160x160x128xbf16>
        %3 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x1x25600x128xbf16>
        %5 = "ttir.reshape"(%1) <{shape = [1 : i32, 10 : i32, 2560 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x10x2560x128xbf16>
        return %3, %5 : tensor<1x1x25600x128xbf16>, tensor<1x10x2560x128xbf16>
    }
}
