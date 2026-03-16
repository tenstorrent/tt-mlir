// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func public @test_topk_datatype_workaround(%arg0: tensor<2x3x32x128xbf16>) -> (tensor<2x3x32x5xbf16>, tensor<2x3x32x5xsi32>) {
    // CHECK-LABEL: func.func public @test_topk_datatype_workaround
    // CHECK: %{{.*}}, %[[INDICES:.*]] = "ttnn.topk"
    // CHECK-SAME: <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = false}>
    // CHECK-SAME: tensor<2x3x32x128xbf16,
    // CHECK-SAME: -> (tensor<2x3x32x5xbf16,
    // CHECK-SAME: tensor<2x3x32x5xui16,
    %values, %indices = "ttir.topk"(%arg0) { k = 5 : i32 } : (tensor<2x3x32x128xbf16>) -> (tensor<2x3x32x5xbf16>, tensor<2x3x32x5xsi32>)
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[INDICES]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<2x3x32x5xui16,
    // CHECK-SAME: -> tensor<2x3x32x5xsi32,
    return %values, %indices : tensor<2x3x32x5xbf16>, tensor<2x3x32x5xsi32>
  }
}
