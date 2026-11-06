// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @empty_index(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.case' op index string must not be empty
    %0 = emitpy.case "" branches {
      emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @placeholder_count_mismatch(%arg0: !emitpy.opaque<"ttnn.Tensor">, %idx: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.case' op requires operands for each placeholder in the index string
    %0 = emitpy.case "{} {}" args %idx : (!emitpy.opaque<"ttnn.Tensor">) branches {
      emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @branch_takes_arguments(%arg0: !emitpy.opaque<"ttnn.Tensor">, %idx: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.case' op expects branch 0 to take no arguments, but it takes 1
    %0 = emitpy.case "{}" args %idx : (!emitpy.opaque<"ttnn.Tensor">) branches {
    ^bb0(%unused: !emitpy.opaque<"ttnn.Tensor">):
      emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @yield_count_mismatch(%arg0: !emitpy.opaque<"ttnn.Tensor">, %idx: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.case' op expected branch 1 to yield 1 values, got 2
    %0 = emitpy.case "{}" args %idx : (!emitpy.opaque<"ttnn.Tensor">) branches {
      emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
    }, {
      emitpy.case_yield %arg0, %arg0 : !emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @yield_type_mismatch(%arg0: !emitpy.opaque<"ttnn.Tensor">, %other: !emitpy.opaque<"int">, %idx: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: error: 'emitpy.case' op value #0 yielded by branch 1 has type '!emitpy.opaque<"int">' but result #0 has type '!emitpy.opaque<"ttnn.Tensor">'
    %0 = emitpy.case "{}" args %idx : (!emitpy.opaque<"ttnn.Tensor">) branches {
      emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
    }, {
      emitpy.case_yield %other : !emitpy.opaque<"int">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @yield_outside_case(%arg0: !emitpy.opaque<"ttnn.Tensor">) {
    // CHECK: error: 'emitpy.case_yield' op expects parent op 'emitpy.case'
    emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
  }
}
