// RUN: ttmlir-opt --ttir-erase-inverse-ops="force=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_commute_permute_reshape_n_1_c_hw(%arg0: tensor<16x7x7x2048xbf16>) -> tensor<16x1x49x2048xbf16> {
        // CHECK-LABEL: func.func @test_commute_permute_reshape_n_1_c_hw
        // CHECK-NOT: ttir.permute
        // CHECK: %[[RESHAPED:[0-9]+]] = "ttir.reshape"(%arg0, %0)
        // CHECK-NOT: "ttir.permute"
        // CHECK: return %[[RESHAPED]]
        %0 = tensor.empty() : tensor<16x2048x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<16x7x7x2048xbf16>, tensor<16x2048x7x7xbf16>) -> tensor<16x2048x7x7xbf16>
        %2 = tensor.empty() : tensor<16x1x2048x49xbf16>
        %3 = "ttir.reshape"(%1, %2) <{shape = [16:i32, 1: i32, 2048: i32, 49: i32]}> : (tensor<16x2048x7x7xbf16>, tensor<16x1x2048x49xbf16>) -> tensor<16x1x2048x49xbf16>
        %4 = tensor.empty() : tensor<16x1x49x2048xbf16>
        %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<16x1x2048x49xbf16>, tensor<16x1x49x2048xbf16>) -> tensor<16x1x49x2048xbf16>
        return %5: tensor<16x1x49x2048xbf16>
    }
    func.func @test_commute_permute_reshape_n_c_1_hw(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1x49x1152xbf16> {
        // CHECK-LABEL: func.func @test_commute_permute_reshape_n_c_1_hw
        // CHECK-NOT: ttir.permute
        // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %0)
        // CHECK-NOT: "ttir.permute"
        // CHECK: return %[[RESHAPE]]
        %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        %2 = ttir.empty() : tensor<12x1152x1x49xbf16>
        %3 = "ttir.reshape"(%1, %2) <{shape = [12 : i32, 1152 : i32, 1 : i32, 49 : i32]}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x49xbf16>) -> tensor<12x1152x1x49xbf16>
        %4 = ttir.empty() : tensor<12x1x49x1152xbf16>
        %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x49xbf16>, tensor<12x1x49x1152xbf16>) -> tensor<12x1x49x1152xbf16>
        return %5 : tensor<12x1x49x1152xbf16>
    }
}
