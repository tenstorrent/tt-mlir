// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
// Test that f32 input to topk is automatically converted to bf16 by the workaround pass

module {
  func.func public @test_topk_f32_input_workaround(%arg0: tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xsi32>) {
    // CHECK-LABEL: func.func public @test_topk_f32_input_workaround
    // CHECK: %[[INPUT_BF16:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<2x3x32x128xf32
    // CHECK-SAME: -> tensor<2x3x32x128xbf16
    // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%[[INPUT_BF16]])
    // CHECK-SAME: <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = false}>
    // CHECK-SAME: tensor<2x3x32x128xbf16
    // CHECK-SAME: -> (tensor<2x3x32x5xbf16
    // CHECK-SAME: tensor<2x3x32x5xui16
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[INDICES]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<2x3x32x5xui16
    // CHECK-SAME: -> tensor<2x3x32x5xsi32
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[VALUES]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: tensor<2x3x32x5xbf16
    // CHECK-SAME: -> tensor<2x3x32x5xf32
    %values, %indices = "ttir.topk"(%arg0) { k = 5 : i32 } : (tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xsi32>)
    return %values, %indices : tensor<2x3x32x5xf32>, tensor<2x3x32x5xsi32>
  }
}
