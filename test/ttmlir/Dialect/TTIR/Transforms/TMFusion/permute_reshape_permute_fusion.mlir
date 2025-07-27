// RUN: ttmlir-opt --ttir-erase-inverse-ops -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_commute_permute(%arg0: tensor<16x7x7x2048xbf16>) -> tensor<16x1x49x2048xbf16> {
        // CHECK-NOT: ttir.permute
        // CHECK: %[[RESHAPED:[0-9]+]] = "ttir.reshape"(%arg0, %0)
        // CHECK: return %[[RESHAPED]]
        %0 = tensor.empty() : tensor<16x2048x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<16x7x7x2048xbf16>, tensor<16x2048x7x7xbf16>) -> tensor<16x2048x7x7xbf16>
        %2 = tensor.empty() : tensor<16x1x2048x49xbf16>
        %3 = "ttir.reshape"(%1, %2) <{shape = [16:i32, 1: i32, 2048: i32, 49: i32]}> : (tensor<16x2048x7x7xbf16>, tensor<16x1x2048x49xbf16>) -> tensor<16x1x2048x49xbf16>
        %4 = tensor.empty() : tensor<16x1x49x2048xbf16>
        %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<16x1x2048x49xbf16>, tensor<16x1x49x2048xbf16>) -> tensor<16x1x49x2048xbf16>
        return %5: tensor<16x1x49x2048xbf16>
    }
}
