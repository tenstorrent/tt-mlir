// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify TTIR -> TTNN lowering of ttir.gated_activation forwards the
// `activation` and `dim` attributes onto ttnn.gated_activation, which the
// runtime dispatches to ::ttnn::glu / swiglu / geglu / reglu.

module {
  // CHECK-LABEL: func.func @forward_swiglu
  // CHECK: "ttnn.gated_activation"
  // CHECK-SAME: activation = "swiglu"
  // CHECK-SAME: dim = -1
  func.func @forward_swiglu(%arg0: tensor<4x64xbf16>) -> tensor<4x32xbf16> {
    %0 = "ttir.gated_activation"(%arg0) <{activation = "swiglu", dim = -1 : si32}> : (tensor<4x64xbf16>) -> tensor<4x32xbf16>
    return %0 : tensor<4x32xbf16>
  }

  // CHECK-LABEL: func.func @forward_glu
  // CHECK: "ttnn.gated_activation"
  // CHECK-SAME: activation = "glu"
  func.func @forward_glu(%arg0: tensor<4x64xbf16>) -> tensor<4x32xbf16> {
    %0 = "ttir.gated_activation"(%arg0) <{activation = "glu", dim = -1 : si32}> : (tensor<4x64xbf16>) -> tensor<4x32xbf16>
    return %0 : tensor<4x32xbf16>
  }

  // CHECK-LABEL: func.func @forward_geglu
  // CHECK: "ttnn.gated_activation"
  // CHECK-SAME: activation = "geglu"
  func.func @forward_geglu(%arg0: tensor<4x64xbf16>) -> tensor<4x32xbf16> {
    %0 = "ttir.gated_activation"(%arg0) <{activation = "geglu", dim = -1 : si32}> : (tensor<4x64xbf16>) -> tensor<4x32xbf16>
    return %0 : tensor<4x32xbf16>
  }

  // CHECK-LABEL: func.func @forward_reglu
  // CHECK: "ttnn.gated_activation"
  // CHECK-SAME: activation = "reglu"
  func.func @forward_reglu(%arg0: tensor<4x64xbf16>) -> tensor<4x32xbf16> {
    %0 = "ttir.gated_activation"(%arg0) <{activation = "reglu", dim = -1 : si32}> : (tensor<4x64xbf16>) -> tensor<4x32xbf16>
    return %0 : tensor<4x32xbf16>
  }
}
