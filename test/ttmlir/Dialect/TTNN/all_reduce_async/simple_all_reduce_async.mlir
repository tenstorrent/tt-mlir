// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @forward(%arg0: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    // CHECK: "ttnn.all_reduce_async"
    %1 = "ttir.all_reduce_async"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }
}
