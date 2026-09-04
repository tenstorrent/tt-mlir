// RUN: ttmlir-opt --canonicalize --ttir-erase-inverse-ops="force=true enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
    func.func @test_commute_identical_users(%arg0: tensor<32x64xbf16>) -> (tensor<64x32xbf16>, tensor<64x32xbf16>) {
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%[[PERMUTE]]
        // CHECK: return %[[EXP]], %[[EXP]]
        %1 = "ttir.exp"(%arg0) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
        %3 = "ttir.permute"(%1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
        %5 = "ttir.permute"(%1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
        return %3, %5 : tensor<64x32xbf16>, tensor<64x32xbf16>
    }
}

module {
    func.func @test_dont_commute_different_users(%arg0: tensor<32x64xbf16>) -> (tensor<64x32xbf16>, tensor<1x2048xbf16>) {
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[EXP]]
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[EXP]]
        // CHECK: return %[[PERMUTE]], %[[RESHAPE]]
        %1 = "ttir.exp"(%arg0) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
        %3 = "ttir.permute"(%1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
        %5 = "ttir.reshape"(%1) <{shape = [1: i32, 2048: i32]}> : (tensor<32x64xbf16>) -> tensor<1x2048xbf16>
        return %3, %5 : tensor<64x32xbf16>, tensor<1x2048xbf16>
    }
}

module {
    func.func @test_commute_reshape(%arg0: tensor<32x64xbf16>) -> tensor<1x2048xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%[[RESHAPE]]
        // CHECK: return %[[EXP]]
        %1 = "ttir.exp"(%arg0) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
        %3 = "ttir.reshape"(%1) <{shape = [1: i32, 2048: i32]}> : (tensor<32x64xbf16>) -> tensor<1x2048xbf16>
        return %3: tensor<1x2048xbf16>
    }
}

module {
    func.func @test_commute_permute(%arg0: tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16> {
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%[[PERMUTE]]
        // CHECK: return %[[EXP]]
        %1 = "ttir.exp"(%arg0) : (tensor<1x3x224x224xbf16>) -> tensor<1x3x224x224xbf16>
        %3 = "ttir.permute"(%1) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16>
        return %3: tensor<1x224x224x3xbf16>
    }
}

// The squeeze version of @test_commute_identical_users, with the permutes replaced by squeezes.
module {
    func.func @commute_identical_squeeze_users(%arg0: tensor<1x32x64xbf16>) -> (tensor<32x64xbf16>, tensor<32x64xbf16>) {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0)
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%[[RESHAPE]]
        // CHECK: return %[[EXP]], %[[EXP]]
        %1 = "ttir.exp"(%arg0) : (tensor<1x32x64xbf16>) -> tensor<1x32x64xbf16>
        %3 = "ttir.squeeze"(%1) <{dim = 0 : si32}> : (tensor<1x32x64xbf16>) -> tensor<32x64xbf16>
        %5 = "ttir.squeeze"(%1) <{dim = 0 : si32}> : (tensor<1x32x64xbf16>) -> tensor<32x64xbf16>
        return %3, %5 : tensor<32x64xbf16>, tensor<32x64xbf16>
    }
}

// The unsqueeze version of @test_commute_reshape.
module {
    func.func @commute_unsqueeze_through_unary(%arg0: tensor<32x64xbf16>) -> tensor<1x32x64xbf16> {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0)
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%[[RESHAPE]]
        // CHECK: return %[[EXP]]
        %1 = "ttir.exp"(%arg0) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
        %3 = "ttir.unsqueeze"(%1) <{dim = 0 : si32}> : (tensor<32x64xbf16>) -> tensor<1x32x64xbf16>
        return %3 : tensor<1x32x64xbf16>
    }
}

// A squeeze on one user and the reshape spelling on the other: both must be seen as the same TM.
module {
    func.func @commute_mixed_squeeze_and_reshape_users(%arg0: tensor<1x32x64xbf16>) -> (tensor<32x64xbf16>, tensor<32x64xbf16>) {
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0)
        // CHECK: %[[SQRT:[0-9]+]] = "ttir.sqrt"(%[[RESHAPE]]
        // CHECK: return %[[SQRT]], %[[SQRT]]
        %1 = "ttir.sqrt"(%arg0) : (tensor<1x32x64xbf16>) -> tensor<1x32x64xbf16>
        %3 = "ttir.squeeze"(%1) <{dim = 0 : si32}> : (tensor<1x32x64xbf16>) -> tensor<32x64xbf16>
        %5 = "ttir.reshape"(%1) <{shape = [32 : i32, 64 : i32]}> : (tensor<1x32x64xbf16>) -> tensor<32x64xbf16>
        return %3, %5 : tensor<32x64xbf16>, tensor<32x64xbf16>
    }
}
