// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @dynamic_update_slice_example() -> (tensor<4xi32>, tensor<4x4xi32>) {
    %base = stablehlo.constant dense<[0, 0, 0, 0]> : tensor<4xi32>
    %update = stablehlo.constant dense<[9, 9]> : tensor<2xi32>
    %start = stablehlo.constant dense<1> : tensor<i32>

    %result = stablehlo.dynamic_update_slice %base, %update, %start
      : (tensor<4xi32>, tensor<2xi32>, tensor<i32>) -> tensor<4xi32>

    %base_2d = stablehlo.constant dense<[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]> : tensor<4x4xi32>
    %update_2d = stablehlo.constant dense<[[9, 9], [9, 9]]> : tensor<2x2xi32>
    %start_2d = stablehlo.constant dense<1> : tensor<i32>

    %result_2d = stablehlo.dynamic_update_slice %base_2d, %update_2d, %start_2d, %start_2d
      : (tensor<4x4xi32>, tensor<2x2xi32>, tensor<i32>, tensor<i32>) -> tensor<4x4xi32>

    return %result, %result_2d : tensor<4xi32>, tensor<4x4xi32>
  }
  // CHECK: ttcore.device_module
  // CHECK-NOT: stablehlo.dynamic_update_slice
  // CHECK: ttcore.cpu_module
  // CHECK-LABEL: func.func @hoisted_stablehlo_dynamic_update_slice_4xi32_2xi32_i32_func
  // CHECK: stablehlo.dynamic_update_slice
  // CHECK-LABEL: func.func @hoisted_stablehlo_dynamic_update_slice_4x4xi32_2x2xi32_i32_i32_func
  // CHECK: stablehlo.dynamic_update_slice
}
