// RUN: ttmlir-opt --ttir-erase-inverse-ops -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test verifies that the EIO pass is skipped when no operations have
// the flattened_compat_info attribute and force flag is not set.
// The TM operations should remain unchanged in the output.

module {
    func.func @test_skip_without_attr(%arg0: tensor<32x64xbf16>) -> tensor<64x32xbf16> {
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%arg0
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[EXP]]
        // CHECK: return %[[PERMUTE]]

        // Without flattened_compat_info and force=false, this permute should NOT
        // be commuted above the exp operation
        %1 = "ttir.exp"(%arg0) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
        %3 = "ttir.permute"(%1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
        return %3 : tensor<64x32xbf16>
    }
}

module {
    func.func @test_inverse_tms_not_erased(%arg0: tensor<1x3x224x224xbf16>) -> tensor<1x3x224x224xbf16> {
        // CHECK: %[[PERMUTE1:[0-9]+]] = "ttir.permute"(%arg0
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%[[PERMUTE1]]
        // CHECK: %[[PERMUTE2:[0-9]+]] = "ttir.permute"(%[[EXP]]
        // CHECK: return %[[PERMUTE2]]

        // These inverse permutations should NOT be erased without flattened_compat_info
        %1 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16>
        %3 = "ttir.exp"(%1) : (tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
        %5 = "ttir.permute"(%3) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x224x224x3xbf16>) -> tensor<1x3x224x224xbf16>
        return %5 : tensor<1x3x224x224xbf16>
    }
}
