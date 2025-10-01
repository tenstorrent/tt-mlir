// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @pow_scalar(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: "ttnn.pow_scalar"(%arg0)
    // CHECK-SAME: {exponent = 2 : i32}
    %1 = "ttir.pow_scalar"(%arg0, %0) <{exponent = 2 : i32}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
