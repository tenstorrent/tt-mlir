// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for EmitPy while op.

module {
  func.func @no_bound(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.while' op requires exactly one of `condition` or `trip_count`
    %0 = emitpy.while inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">):
      emitpy.while_yield %acc : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @both_bounds(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.while' op requires exactly one of `condition` or `trip_count`
    %0 = emitpy.while "True" trip_count 4 inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">):
      emitpy.while_yield %acc : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @empty_condition(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.while' op condition string must not be empty
    %0 = emitpy.while "" inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">):
      emitpy.while_yield %acc : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @placeholder_count_mismatch(%arg0: !emitpy.opaque<"ttnn.Tensor">,
                                        %arg1: !emitpy.opaque<"int">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.while' op requires operands for each placeholder in the condition string
    %0 = emitpy.while "{} > {}" args %arg1 : (!emitpy.opaque<"int">)
        inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">):
      emitpy.while_yield %acc : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @block_arg_count_mismatch(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.while' op expected 1 body block arguments to match its loop-carried values, got 2
    %0 = emitpy.while trip_count 4 inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">, %extra: !emitpy.opaque<"ttnn.Tensor">):
      emitpy.while_yield %acc : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @block_arg_type_mismatch(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.while' op body block arguments #0 has type '!emitpy.opaque<"int">' but init #0 has type '!emitpy.opaque<"ttnn.Tensor">'
    %0 = emitpy.while trip_count 4 inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"int">):
      emitpy.while_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @result_count_mismatch(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.while' op expected 1 results to match its loop-carried values, got 2
    %0:2 = emitpy.while trip_count 4 inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">):
      emitpy.while_yield %acc : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">)
    return %0#0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @yield_count_mismatch(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.while' op expected 1 yielded values to match its loop-carried values, got 2
    %0 = emitpy.while trip_count 4 inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">):
      emitpy.while_yield %acc, %acc : !emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @yield_outside_while(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> () {
    // CHECK: error: 'emitpy.while_yield' op expects parent op 'emitpy.while'
    emitpy.while_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
  }
}
