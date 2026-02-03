// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func public @topk_custom_call(%arg0: tensor<4x16xbf16>) -> (tensor<4x8xbf16>, tensor<4x8xi32>) {
    // CHECK-LABEL: func.func public @topk_custom_call(
    // CHECK: %[[VAL:[0-9]+]], %[[IDX:[0-9]+]] = "ttir.topk"(%arg0)
    // CHECK-SAME: <{dim = -1 : si32, k = 8 : ui32, largest = true, sorted = true}>
    // CHECK-SAME: (tensor<4x16xbf16>) -> (tensor<4x8xbf16>, tensor<4x8xi32>)
    %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {backend_config = "{\"k\": 8, \"dim\": -1, \"largest\": true, \"sorted\": true}"} : (tensor<4x16xbf16>) -> (tensor<4x8xbf16>, tensor<4x8xi32>)
    return %0#0, %0#1 : tensor<4x8xbf16>, tensor<4x8xi32>
  }
}
