// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-upwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
    func.func @test_reshape_slice_commute_downwards(%arg0: tensor<1x1x25600x128xbf16>) -> tensor<1x160x160x64xbf16> {
        // CHECK: %[[SLICE:[0-9]+]] = "ttir.slice_static"(%arg0,
        // CHECK: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
        // CHECK: ends = [1 : i32, 1 : i32, 25600 : i32, 64 : i32]
        // CHECK: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SLICE]]
        // CHECK: shape = [1 : i32, 160 : i32, 160 : i32, 64 : i32]
        // CHECK: return %[[RESHAPE]]
        %0 = ttir.empty() : tensor<1x160x160x128xbf16>
        %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]}> : (tensor<1x1x25600x128xbf16>, tensor<1x160x160x128xbf16>) -> tensor<1x160x160x128xbf16>
        %2 = ttir.empty() : tensor<1x160x160x64xbf16>
        %3 = "ttir.slice_static"(%1, %2) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<1x160x160x64xbf16>) -> tensor<1x160x160x64xbf16>
        return %3 : tensor<1x160x160x64xbf16>
    }
    // If slice is done on a dim left to the rightmost reshape dim, they cannot commute.
    func.func @test_slice_reshape_no_commute_overlapping_dims(%arg0: tensor<1x160x160x128xbf16>) -> tensor<160x80x1x64xbf16> {
        // CHECK: "ttir.slice_static"
        // CHECK: "ttir.reshape"
        %0 = ttir.empty() : tensor<1x80x160x64xbf16>
        %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0 : i32, 80 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<1x80x160x64xbf16>) -> tensor<1x80x160x64xbf16>
        %2 = ttir.empty() : tensor<160x80x1x64xbf16>
        %3 = "ttir.reshape"(%1, %2) <{shape = [160 : i32, 80 : i32, 1 : i32, 64 : i32]}> : (tensor<1x80x160x64xbf16>, tensor<160x80x1x64xbf16>) -> tensor<160x80x1x64xbf16>
        return %3 : tensor<160x80x1x64xbf16>
    }
    func.func @test_reshape_slice_commute_downwards_dim_dec(%arg0: tensor<25600x128xbf16>) -> tensor<1x160x160x64xbf16> {
        // CHECK: %[[SLICE:[0-9]+]] = "ttir.slice_static"(%arg0,
        // CHECK: begins = [0 : i32, 0 : i32]
        // CHECK: ends = [25600 : i32, 64 : i32]
        // CHECK: step = [1 : i32, 1 : i32]
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SLICE]]
        // CHECK: shape = [1 : i32, 160 : i32, 160 : i32, 64 : i32]
        // CHECK: return %[[RESHAPE]]
        %0 = ttir.empty() : tensor<1x160x160x128xbf16>
        %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]}> : (tensor<25600x128xbf16>, tensor<1x160x160x128xbf16>) -> tensor<1x160x160x128xbf16>
        %2 = ttir.empty() : tensor<1x160x160x64xbf16>
        %3 = "ttir.slice_static"(%1, %2) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<1x160x160x64xbf16>) -> tensor<1x160x160x64xbf16>
        return %3 : tensor<1x160x160x64xbf16>
    }
    func.func @test_reshape_slice_commute_downwards_dim_inc(%arg0: tensor<1x160x160x128xbf16>) -> tensor<25600x64xbf16> {
         // CHECK: %[[SLICE:[0-9]+]] = "ttir.slice_static"(%arg0,
         // CHECK: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
         // CHECK: ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32]
         // CHECK: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
         // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[SLICE]]
         // CHECK: shape = [25600 : i32, 64 : i32]
         // CHECK: return %[[RESHAPE]]
        %0 = ttir.empty() : tensor<25600x128xbf16>
        %1 = "ttir.reshape"(%arg0, %0) <{shape = [25600 : i32, 128 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<25600x128xbf16>) -> tensor<25600x128xbf16>
        %2 = ttir.empty() : tensor<25600x64xbf16>
        %3 = "ttir.slice_static"(%1, %2) <{begins = [0 : i32, 0 : i32], ends = [25600 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<25600x128xbf16>, tensor<25600x64xbf16>) -> tensor<25600x64xbf16>
        return %3 : tensor<25600x64xbf16>
    }
    func.func @test_reshape_slice_commute_downwards_squeeze(%arg0: tensor<1x1x25600x128xbf16>) -> tensor<25600x64xbf16> {
        // CHECK: "ttir.slice_static"
        // CHECK: "ttir.reshape"
        %0 = ttir.empty() : tensor<25600x128xbf16>
        %1 = "ttir.reshape"(%arg0, %0) <{shape = [25600 : i32, 128 : i32]}> : (tensor<1x1x25600x128xbf16>, tensor<25600x128xbf16>) -> tensor<25600x128xbf16>
        %2 = ttir.empty() : tensor<25600x64xbf16>
        %3 = "ttir.slice_static"(%1, %2) <{begins = [0 : i32, 0 : i32], ends = [25600 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<25600x128xbf16>, tensor<25600x64xbf16>) -> tensor<25600x64xbf16>
        return %3 : tensor<25600x64xbf16>
    }
    func.func @test_reshape_slice_commute_downwards_unsqueeze(%arg0: tensor<25600x128xbf16>) -> tensor<1x1x25600x64xbf16> {
        // CHECK: "ttir.slice_static"
        // CHECK: "ttir.reshape"
        %0 = ttir.empty() : tensor<1x1x25600x128xbf16>
        %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 1 : i32, 25600 : i32, 128 : i32]}> : (tensor<25600x128xbf16>, tensor<1x1x25600x128xbf16>) -> tensor<1x1x25600x128xbf16>
        %2 = ttir.empty() : tensor<1x1x25600x64xbf16>
        %3 = "ttir.slice_static"(%1, %2) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 25600 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x25600x128xbf16>, tensor<1x1x25600x64xbf16>) -> tensor<1x1x25600x64xbf16>
        return %3 : tensor<1x1x25600x64xbf16>
    }
    // If reshape has multiple users, it does not commute downwards.
    func.func @test_reshape_slice_no_commute_multiple_users(%arg0: tensor<25600x128xbf16>) -> (tensor<1x160x160x64xbf16>, tensor<1x160x160x64xbf16>) {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0
        // CHECK: %[[SLICE:[0-9]+]] = "ttir.slice_static"(%[[RESHAPE]]
        // CHECK: %[[SLICE2:[0-9]+]] = "ttir.slice_static"(%[[RESHAPE]]
        // CHECK: return %[[SLICE]], %[[SLICE2]]
        %0 = ttir.empty() : tensor<1x160x160x128xbf16>
        %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 160 : i32, 160 : i32, 128 : i32]}> : (tensor<25600x128xbf16>, tensor<1x160x160x128xbf16>) -> tensor<1x160x160x128xbf16>
        %2 = ttir.empty() : tensor<1x160x160x64xbf16>
        %3 = "ttir.slice_static"(%1, %2) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<1x160x160x64xbf16>) -> tensor<1x160x160x64xbf16>
        %4 = ttir.empty() : tensor<1x160x160x64xbf16>
        %5 = "ttir.slice_static"(%1, %4) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<1x160x160x64xbf16>) -> tensor<1x160x160x64xbf16>
        return %3, %5 : tensor<1x160x160x64xbf16>, tensor<1x160x160x64xbf16>
    }
}
