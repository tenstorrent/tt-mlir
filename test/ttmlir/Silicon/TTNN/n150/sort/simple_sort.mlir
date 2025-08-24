// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>) {
    // CHECK-LABEL: @test_sort
    %0 = ttir.empty() : tensor<64x128xbf16>
    %1 = ttir.empty() : tensor<64x128xi16>
    // CHECK: %{{.*}}, %{{.*}} = "ttnn.sort"(%arg0)
    // CHECK-SAME: <{descending = false, dim = -1 : si8, stable = false}>
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: -> (tensor<64x128xbf16,
    // CHECK-SAME: tensor<64x128xui16,
    %2, %3 = "ttir.sort"(%arg0, %0, %1) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xi16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>)
    return %2, %3 : tensor<64x128xbf16>, tensor<64x128xi16>
  }
}
