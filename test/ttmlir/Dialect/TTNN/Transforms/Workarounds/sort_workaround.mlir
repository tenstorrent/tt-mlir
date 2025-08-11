// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func public @test_sort_datatype_workaround(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xsi32>) {
    %0 = ttir.empty() : tensor<64x128xbf16>
    %1 = ttir.empty() : tensor<64x128xsi32>
    // CHECK-LABEL: func.func public @test_sort_datatype_workaround
    // CHECK: %{{.*}}, %[[INDICES:.*]] = "ttnn.sort"
    // CHECK-SAME: <{descending = false, dim = -1 : si8, stable = false}>
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: -> (tensor<64x128xbf16,
    // CHECK-SAME: tensor<64x128xui16,
    %2, %3 = "ttir.sort"(%arg0, %0, %1) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xsi32>) -> (tensor<64x128xbf16>, tensor<64x128xsi32>)
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[INDICES]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<64x128xui16,
    // CHECK-SAME: -> tensor<64x128xsi32,
    return %2, %3 : tensor<64x128xbf16>, tensor<64x128xsi32>
  }
}
