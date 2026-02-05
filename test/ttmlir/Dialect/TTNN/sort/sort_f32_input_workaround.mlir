// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s | FileCheck %s
// Test that f32 input to sort is automatically converted to bf16

module attributes {} {
  func.func @test_sort_f32_to_bf16_workaround(%arg0: tensor<1x10xf32>) -> (tensor<1x10xf32>, tensor<1x10xi32>) {
    // CHECK-LABEL: @test_sort_f32_to_bf16_workaround
    // CHECK: %[[TYPECAST:.*]] = "ttnn.typecast"(%arg0)
    // CHECK-SAME: <{dtype = #ttcore.supportedDataTypes<bf16>}>
    // CHECK-SAME: (tensor<1x10xf32,
    // CHECK-SAME: -> tensor<1x10xbf16,
    // CHECK: %{{.*}}, %{{.*}} = "ttnn.sort"(%[[TYPECAST]])
    // CHECK-SAME: <{descending = true, dim = 1 : si8, stable = false}>
    // CHECK-SAME: (tensor<1x10xbf16,
    // CHECK-SAME: -> (tensor<1x10xbf16,
    // CHECK-SAME: tensor<1x10xui16,
    %values, %indices = "ttir.sort"(%arg0) <{descending = true, dim = 1 : si32, stable = false}> : (tensor<1x10xf32>) -> (tensor<1x10xf32>, tensor<1x10xi32>)
    return %values, %indices : tensor<1x10xf32>, tensor<1x10xi32>
  }
}
