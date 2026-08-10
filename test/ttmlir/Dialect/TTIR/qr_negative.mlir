// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Verifier negative tests for `ttir.qr`. The op requires statically known
// input, Q, and R shapes: the reduced QR shapes are derived from the input
// ([m, k] and [k, n] with k = min(m, n)), so a dynamic dimension on any of
// the three tensors cannot be validated.

// Dynamic input: verifier must reject the op.
module {
  func.func @qr_dynamic_input(%arg0: tensor<?x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>) {
    // expected-error @+1 {{requires statically known input, Q, and R shapes}}
    %0:2 = "ttir.qr"(%arg0) : (tensor<?x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)
    return %0#0, %0#1 : tensor<3x3xf32>, tensor<3x3xf32>
  }
}

// -----

// Dynamic results: verifier must reject the op.
module {
  func.func @qr_dynamic_results(%arg0: tensor<4x3xf32>) -> (tensor<?x3xf32>, tensor<?x3xf32>) {
    // expected-error @+1 {{requires statically known input, Q, and R shapes}}
    %0:2 = "ttir.qr"(%arg0) : (tensor<4x3xf32>) -> (tensor<?x3xf32>, tensor<?x3xf32>)
    return %0#0, %0#1 : tensor<?x3xf32>, tensor<?x3xf32>
  }
}
