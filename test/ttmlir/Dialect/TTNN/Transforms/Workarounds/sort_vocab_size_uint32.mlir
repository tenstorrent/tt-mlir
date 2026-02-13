// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func public @test_sort_vocab_size_uint32(%arg0: tensor<1x50272xbf16>) -> (tensor<1x50272xbf16>, tensor<1x50272xsi32>) {
    // CHECK-LABEL: func.func public @test_sort_vocab_size_uint32
    // CHECK: %{{.*}}, %[[INDICES:.*]] = "ttnn.sort"
    // CHECK-SAME: <{descending = false, dim = -1 : si8, stable = false}>
    // CHECK-SAME: tensor<1x50272xbf16,
    // CHECK-SAME: -> (tensor<1x50272xbf16,
    // CHECK-SAME: tensor<1x50272xui32,
    %0, %1 = "ttir.sort"(%arg0) : (tensor<1x50272xbf16>) -> (tensor<1x50272xbf16>, tensor<1x50272xsi32>)
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[INDICES]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<1x50272xui32,
    // CHECK-SAME: -> tensor<1x50272xsi32,
    return %0, %1 : tensor<1x50272xbf16>, tensor<1x50272xsi32>
  }
}
