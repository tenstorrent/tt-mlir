// RUN: ttmlir-opt --convert-chlo-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHLO inverse-trig and error-function ops map 1:1 onto the existing TTIR ops,
// so --convert-chlo-to-ttir should rewrite each one directly without
// decomposing it into many primitive ops.
module {
  // CHECK-LABEL: func.func @chlo_acos
  func.func @chlo_acos(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    // CHECK: "ttir.acos"
    // CHECK-NOT: chlo.acos
    %0 = "chlo.acos"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }

  // CHECK-LABEL: func.func @chlo_asin
  func.func @chlo_asin(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    // CHECK: "ttir.asin"
    %0 = "chlo.asin"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }

  // CHECK-LABEL: func.func @chlo_atan
  func.func @chlo_atan(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    // CHECK: "ttir.atan"
    %0 = "chlo.atan"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }

  // CHECK-LABEL: func.func @chlo_erf
  func.func @chlo_erf(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    // CHECK: "ttir.erf"
    %0 = "chlo.erf"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }

  // CHECK-LABEL: func.func @chlo_erfc
  func.func @chlo_erfc(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    // CHECK: "ttir.erfc"
    %0 = "chlo.erfc"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
}
