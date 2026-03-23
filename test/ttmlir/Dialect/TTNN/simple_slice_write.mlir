// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @slice_write(%arg0: tensor<1x197x768xbf16>, %arg1: tensor<1x1x768xbf16>, %arg2: tensor<3xi32>) -> tensor<1x197x768xbf16> {
    // CHECK: "ttnn.slice_write"
    %1 = "ttir.slice_write"(%arg0, %arg1, %arg2) : (tensor<1x197x768xbf16>, tensor<1x1x768xbf16>, tensor<3xi32>) -> tensor<1x197x768xbf16>
    return %1 : tensor<1x197x768xbf16>
  }
}
