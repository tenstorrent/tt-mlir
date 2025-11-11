// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @dynamic_update_slice_example() -> tensor<4xi32> {
    %base = stablehlo.constant dense<[0, 0, 0, 0]> : tensor<4xi32>
    %update = stablehlo.constant dense<[9, 9]> : tensor<2xi32>
    %start = stablehlo.constant dense<1> : tensor<i32>

    %result = stablehlo.dynamic_update_slice %base, %update, %start
      : (tensor<4xi32>, tensor<2xi32>, tensor<i32>) -> tensor<4xi32>

    return %result : tensor<4xi32>
  }
  // CHECK: ttcore.device_module
  // CHECK-NOT: stablehlo.dynamic_update_slice
  // CHECK: ttcore.cpu_module
  // CHECK: stablehlo.dynamic_update_slice
}
