// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Loop-carried types must be invariant across the back-edge: the runtime binds
// the values yielded by the body to the region's inputs on the next iteration.
func.func @body_changes_type(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // expected-error @+1 {{value 0 yielded by the 'body' region has type 'tensor<1xf32>' but init 0 has type 'tensor<1xi32>'}}
  %r = ttir.while inits(%arg0 : tensor<1xi32>)
    cond {
    ^cond(%i: tensor<1xi32>):
      %p = "ttir.lt"(%i, %i) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      ttir.yield %p : tensor<1xi1>
    } do {
    ^body(%i: tensor<1xi32>):
      %c = "ttir.typecast"(%i) : (tensor<1xi32>) -> tensor<1xf32>
      ttir.yield %c : tensor<1xf32>
    } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

// The regions observe `inits ++ captures`, so a mismatched signature is an
// error rather than something the serializer would silently misinterpret.
func.func @missing_capture_argument(%arg0: tensor<1xi32>, %cap: tensor<1xi32>) -> tensor<1xi32> {
  // expected-error @+1 {{expects the 'cond' region to take 2 arguments (inits followed by captures), but it takes 1}}
  %r = ttir.while inits(%arg0 : tensor<1xi32>) captures(%cap : tensor<1xi32>)
    cond {
    ^cond(%i: tensor<1xi32>):
      %p = "ttir.lt"(%i, %i) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      ttir.yield %p : tensor<1xi1>
    } do {
    ^body(%i: tensor<1xi32>, %c: tensor<1xi32>):
      ttir.yield %i : tensor<1xi32>
    } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

// The condition must reduce to a single element the runtime can read back.
func.func @multi_element_condition(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // expected-error @+1 {{expects the 'cond' region to yield a single-element tensor}}
  %r = ttir.while inits(%arg0 : tensor<1xi32>)
    cond {
    ^cond(%i: tensor<1xi32>):
      %b = "ttir.broadcast"(%i) <{broadcast_dimensions = array<i64: 4>}> : (tensor<1xi32>) -> tensor<4xi32>
      %p = "ttir.lt"(%b, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
      ttir.yield %p : tensor<4xi1>
    } do {
    ^body(%i: tensor<1xi32>):
      ttir.yield %i : tensor<1xi32>
    } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

// ttir.while is IsolatedFromAbove: each region becomes its own runtime program
// with its own tensor pool, so anything it reads has to arrive as a capture.
func.func @unpromoted_capture(%arg0: tensor<1xi32>, %outside: tensor<1xi32>) -> tensor<1xi32> {
  // expected-note @+1 {{required by region isolation constraints}}
  %r = ttir.while inits(%arg0 : tensor<1xi32>)
    cond {
    ^cond(%i: tensor<1xi32>):
      // expected-error @+1 {{using value defined outside the region}}
      %p = "ttir.lt"(%i, %outside) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      ttir.yield %p : tensor<1xi1>
    } do {
    ^body(%i: tensor<1xi32>):
      ttir.yield %i : tensor<1xi32>
    } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}
