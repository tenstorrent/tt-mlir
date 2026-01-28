// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-upwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
    func.func @test_reshape_softmax_commute_downwards_flatten(%arg0: tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16> {
        // CHECK: %[[SOFTMAX:[0-9]+]] = "ttir.softmax"(%arg0
        // CHECK: dimension = 3
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SOFTMAX]]
        // CHECK: shape = [1 : i32, 25600 : i32, 128 : i32]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16>
        %2 = "ttir.softmax"(%1) <{dimension = 2 : si32}> : (tensor<1x25600x128xbf16>) -> tensor<1x25600x128xbf16>
        return %2 : tensor<1x25600x128xbf16>
    }
    func.func @test_reshape_softmax_commute_downwards_unflatten(%arg0: tensor<1x25600x128xbf16>) -> tensor<1x160x160x128xbf16> {
        // CHECK: %[[SOFTMAX:[0-9]+]] = "ttir.softmax"(%arg0
        // CHECK: dimension = 2
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SOFTMAX]]
        // CHECK: shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]}> : (tensor<1x25600x128xbf16>) -> tensor<1x160x160x128xbf16>
        %2 = "ttir.softmax"(%1) <{dimension = 3 : si32}> : (tensor<1x160x160x128xbf16>) -> tensor<1x160x160x128xbf16>
        return %2 : tensor<1x160x160x128xbf16>

    }
    func.func @test_reshape_softmax_commute_downwards_split_on_softmax_dim(%arg0: tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: shape = [1 : i32, 25600 : i32, 128 : i32]
        // CHECK: %[[SOFTMAX:[0-9]+]] = "ttir.softmax"(%[[RESHAPE]]
        // CHECK: dimension = 1
        // CHECK: return %[[SOFTMAX]]
        %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16>
        %2 = "ttir.softmax"(%1) <{dimension = 1 : si32}> : (tensor<1x25600x128xbf16>) -> tensor<1x25600x128xbf16>
        return %2 : tensor<1x25600x128xbf16>
    }
    func.func @test_reshape_softmax_commute_downwards_merge_on_softmax_dim(%arg0: tensor<1x25600x128xbf16>) -> tensor<1x160x160x128xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]
        // CHECK: %[[SOFTMAX:[0-9]+]] = "ttir.softmax"(%[[RESHAPE]]
        // CHECK: dimension = 2
        // CHECK: return %[[SOFTMAX]]
        %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]}> : (tensor<1x25600x128xbf16>) -> tensor<1x160x160x128xbf16>
        %2 = "ttir.softmax"(%1) <{dimension = 2 : si32}> : (tensor<1x160x160x128xbf16>) -> tensor<1x160x160x128xbf16>
        return %2 : tensor<1x160x160x128xbf16>
    }
    func.func @test_reshape_softmax_commute_downwards_negative_softmax_dim(%arg0: tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16> {
        // CHECK: %[[SOFTMAX:[0-9]+]] = "ttir.softmax"(%arg0
        // CHECK: dimension = 3
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SOFTMAX]]
        // CHECK: shape = [1 : i32, 25600 : i32, 128 : i32]
        // CHECK: return %[[RESHAPE]]
        %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>) -> tensor<1x25600x128xbf16>
        %2 = "ttir.softmax"(%1) <{dimension = -1 : si32}> : (tensor<1x25600x128xbf16>) -> tensor<1x25600x128xbf16>
        return %2 : tensor<1x25600x128xbf16>
    }
}
