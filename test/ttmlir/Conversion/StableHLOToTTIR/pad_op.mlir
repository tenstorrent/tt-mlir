// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_module_pad {
  func.func public @test_pad(%arg0: tensor<5x10xbf16>, %arg1: tensor<bf16>) -> tensor<7x11xbf16> {
    // CHECK-LABEL: func.func public @test_pad
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty
    // CHECK: %[[VAL:[0-9]+]] = "ttir.pad"(%arg0, %arg1, %[[EMPTY]])
    // CHECK-SAME: padding = array<i32: 1, 0, 1, 1>
    // CHECK-NOT: interior =
    // CHECK-SAME: (tensor<5x10xbf16>, tensor<1xbf16>, tensor<7x11xbf16>) -> tensor<7x11xbf16>
    // CHECK: return %[[VAL]]
    %0 = stablehlo.pad %arg0, %arg1, low = [1, 0], high = [1, 1], interior = [0, 0] : (tensor<5x10xbf16>, tensor<bf16>) -> tensor<7x11xbf16>
    return %0 : tensor<7x11xbf16>
  }
}
