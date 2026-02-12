// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
// Test that f32 input to sort is automatically converted to bf16 by the workaround pass

module {
  func.func public @test_sort_f32_input_workaround(%arg0: tensor<1x10xf32>) -> (tensor<1x10xf32>, tensor<1x10xsi32>) {
    // CHECK-LABEL: func.func public @test_sort_f32_input_workaround
    // CHECK: %[[INPUT_BF16:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<1x10xf32
    // CHECK-SAME: -> tensor<1x10xbf16
    // CHECK: %[[VALUES:.*]], %{{.*}} = "ttnn.sort"(%[[INPUT_BF16]])
    // CHECK-SAME: <{descending = true, dim = 1 : si8, stable = false}>
    // CHECK-SAME: tensor<1x10xbf16
    // CHECK-SAME: -> (tensor<1x10xbf16
    // CHECK-SAME: tensor<1x10xui16
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[VALUES]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: tensor<1x10xbf16
    // CHECK-SAME: -> tensor<1x10xf32
    %values, %indices = "ttir.sort"(%arg0) <{descending = true, dim = 1 : si32, stable = false}> : (tensor<1x10xf32>) -> (tensor<1x10xf32>, tensor<1x10xsi32>)
    return %values, %indices : tensor<1x10xf32>, tensor<1x10xsi32>
  }
}
