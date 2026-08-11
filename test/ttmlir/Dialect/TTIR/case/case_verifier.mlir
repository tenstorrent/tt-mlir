// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// A branch runs at most once, so its block arguments are exactly the captures -
// there are no carried values to prepend.
func.func @missing_capture_argument(%index: tensor<i32>, %cap: tensor<1xi32>) -> tensor<1xi32> {
  // expected-error @+1 {{expects branch 0 to take 1 arguments (the captures), but it takes 0}}
  %r = ttir.case index(%index : tensor<i32>) captures(%cap : tensor<1xi32>)
  branches {
  ^bb0():
    ttir.yield %cap : tensor<1xi32>
  } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

func.func @branch_argument_type_mismatch(%index: tensor<i32>, %cap: tensor<1xi32>) -> tensor<1xi32> {
  // expected-error @+1 {{argument 0 of branch 1 has type 'tensor<1xf32>' but 'tensor<1xi32>' was expected}}
  %r = ttir.case index(%index : tensor<i32>) captures(%cap : tensor<1xi32>)
  branches {
  ^bb0(%c: tensor<1xi32>):
    ttir.yield %c : tensor<1xi32>
  }, {
  ^bb0(%c: tensor<1xf32>):
    ttir.yield %cap : tensor<1xi32>
  } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

// Every branch feeds the one set of results, so they all yield one value per
// result.
func.func @branch_yield_count_mismatch(%index: tensor<i32>, %cap: tensor<1xi32>) -> tensor<1xi32> {
  // expected-error @+1 {{expects branch 1 to yield one value per result (1), but it yields 2}}
  %r = ttir.case index(%index : tensor<i32>) captures(%cap : tensor<1xi32>)
  branches {
  ^bb0(%c: tensor<1xi32>):
    ttir.yield %c : tensor<1xi32>
  }, {
  ^bb0(%c: tensor<1xi32>):
    ttir.yield %c, %c : tensor<1xi32>, tensor<1xi32>
  } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

// Comparing types exactly is what makes the branches agree on layout once they
// carry a TTNNLayoutAttr encoding; the consumer reads one descriptor for all of
// them.
func.func @branch_yield_type_mismatch(%index: tensor<i32>, %cap: tensor<1xi32>) -> tensor<1xi32> {
  // expected-error @+1 {{value 0 yielded by branch 0 has type 'tensor<1xf32>' but result 0 has type 'tensor<1xi32>'}}
  %r = ttir.case index(%index : tensor<i32>) captures(%cap : tensor<1xi32>)
  branches {
  ^bb0(%c: tensor<1xi32>):
    %f = "ttir.typecast"(%c) : (tensor<1xi32>) -> tensor<1xf32>
    ttir.yield %f : tensor<1xf32>
  }, {
  ^bb0(%c: tensor<1xi32>):
    ttir.yield %c : tensor<1xi32>
  } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

// The runtime reads the index back to host as an integer.
func.func @non_integer_index(%index: tensor<f32>, %cap: tensor<1xi32>) -> tensor<1xi32> {
  // expected-error @+1 {{expects an integer index tensor, but got 'tensor<f32>'}}
  %r = ttir.case index(%index : tensor<f32>) captures(%cap : tensor<1xi32>)
  branches {
  ^bb0(%c: tensor<1xi32>):
    ttir.yield %c : tensor<1xi32>
  } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

// A branch is picked from one scalar, so a multi-element index has no meaning.
func.func @multi_element_index(%index: tensor<4xi32>, %cap: tensor<1xi32>) -> tensor<1xi32> {
  // expected-error @+1 {{expects a single-element index tensor, but got 'tensor<4xi32>'}}
  %r = ttir.case index(%index : tensor<4xi32>) captures(%cap : tensor<1xi32>)
  branches {
  ^bb0(%c: tensor<1xi32>):
    ttir.yield %c : tensor<1xi32>
  } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

// The op is IsolatedFromAbove, so a branch may not read the enclosing scope
// directly: the value has to be promoted to a capture.
func.func @unpromoted_capture(%index: tensor<i32>, %cap: tensor<1xi32>) -> tensor<1xi32> {
  // expected-note @+1 {{required by region isolation constraints}}
  %r = ttir.case index(%index : tensor<i32>)
  branches {
  ^bb0():
    // expected-error @+1 {{using value defined outside the region}}
    %0 = "ttir.abs"(%cap) : (tensor<1xi32>) -> tensor<1xi32>
    ttir.yield %0 : tensor<1xi32>
  } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}
