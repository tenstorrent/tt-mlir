// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-downwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
    func.func @test_reshape_slice_basic(%arg0: tensor<1x160x160x128xbf16>) -> tensor<1x1x25600x64xbf16> {
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.slice_static"
        %0 = ttir.empty() : tensor<1x160x160x64xbf16>
        %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<1x160x160x64xbf16>) -> tensor<1x160x160x64xbf16>
        %2 = ttir.empty() : tensor<1x1x25600x64xbf16>
        %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 1 : i32, 25600 : i32, 64 : i32]}> : (tensor<1x160x160x64xbf16>, tensor<1x1x25600x64xbf16>) -> tensor<1x1x25600x64xbf16>
        return %3 : tensor<1x1x25600x64xbf16>
    }
    func.func @test_reshape_two_slices(%arg0: tensor<1x160x160x128xbf16>) -> (tensor<1x1x25600x64xbf16>, tensor<1x1x25600x64xbf16>) {
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.slice_static"
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.slice_static"
        %0 = ttir.empty() : tensor<1x160x160x64xbf16>
        %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<1x160x160x64xbf16>) -> tensor<1x160x160x64xbf16>
        %2 = ttir.empty() : tensor<1x160x160x64xbf16>
        %3 = "ttir.slice_static"(%arg0, %2) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<1x160x160x64xbf16>) -> tensor<1x160x160x64xbf16>
        %4 = ttir.empty() : tensor<1x1x25600x64xbf16>
        %5 = "ttir.reshape"(%1, %4) <{shape = [1 : i32, 1 : i32, 25600 : i32, 64 : i32]}> : (tensor<1x160x160x64xbf16>, tensor<1x1x25600x64xbf16>) -> tensor<1x1x25600x64xbf16>
        %6 = ttir.empty() : tensor<1x1x25600x64xbf16>
        %7 = "ttir.reshape"(%3, %6) <{shape = [1 : i32, 1 : i32, 25600 : i32, 64 : i32]}> : (tensor<1x160x160x64xbf16>, tensor<1x1x25600x64xbf16>) -> tensor<1x1x25600x64xbf16>
        return %5, %7 : tensor<1x1x25600x64xbf16>, tensor<1x1x25600x64xbf16>
    }

    func.func @test_reshape_slice_commute_upwards_dim_dec(%arg0: tensor<1x160x160x128xbf16>) -> tensor<25600x64xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: shape = [25600 : i32, 128 : i32]
        // CHECK: %[[SLICE:[0-9]+]] = "ttir.slice_static"(%[[RESHAPE]]
        // CHECK: begins = [0 : i32, 0 : i32]
        // CHECK: ends = [25600 : i32, 64 : i32]
        // CHECK: step = [1 : i32, 1 : i32]
        // CHECK: return %[[SLICE]]
        %0 = ttir.empty() : tensor<1x160x160x64xbf16>
        %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<1x160x160x64xbf16>) -> tensor<1x160x160x64xbf16>
        %2 = ttir.empty() : tensor<25600x64xbf16>
        %3 = "ttir.reshape"(%1, %2) <{shape = [25600 : i32, 64 : i32]}> : (tensor<1x160x160x64xbf16>, tensor<25600x64xbf16>) -> tensor<25600x64xbf16>
        return %3 : tensor<25600x64xbf16>
    }
    func.func @test_slice_reshape_commute_upwards_dim_inc(%arg0: tensor<25600x128xbf16>) -> tensor<1x160x160x64xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]
        // CHECK: %[[SLICE:[0-9]+]] = "ttir.slice_static"(%[[RESHAPE]]
        // CHECK: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
        // CHECK: ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32]
        // CHECK: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
        // CHECK: return %[[SLICE]]
        %0 = ttir.empty() : tensor<25600x64xbf16>
        %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0 : i32, 0 : i32], ends = [25600 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<25600x128xbf16>, tensor<25600x64xbf16>) -> tensor<25600x64xbf16>
        %2 = ttir.empty() : tensor<1x160x160x64xbf16>
        %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 160 : i32, 160 : i32, 64 : i32]}> : (tensor<25600x64xbf16>, tensor<1x160x160x64xbf16>) -> tensor<1x160x160x64xbf16>
        return %3 : tensor<1x160x160x64xbf16>
    }
}
