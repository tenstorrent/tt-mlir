// RUN: ttmlir-opt --ttir-erase-inverse-ops -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test verifies that the EIO pass DOES run when at least one operation
// has the flattened_compat_info attribute. The inverse TM operations should
// be erased in the output.

module {
    func.func @test_inverse_tms_erased(%arg0: tensor<1x3x224x224xbf16>) -> tensor<1x3x224x224xbf16> {
        // CHECK-NOT: "ttir.permute"
        // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%arg0
        // CHECK: return %[[EXP]]

        // These inverse permutations should be erased with flattened_compat_info
        %1 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16>
        %3 = "ttir.exp"(%1) {flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 224, input_width = 224>} : (tensor<1x224x224x3xbf16>) -> tensor<1x224x224x3xbf16>
        %5 = "ttir.permute"(%3) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x224x224x3xbf16>) -> tensor<1x3x224x224xbf16>
        return %5 : tensor<1x3x224x224xbf16>
    }
}
