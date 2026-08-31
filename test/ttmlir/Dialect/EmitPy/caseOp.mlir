// RUN: ttmlir-opt --split-input-file %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: func.func @two_branch
func.func @two_branch(%arg0: !emitpy.opaque<"ttnn.Tensor">, %idx: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
  // CHECK: emitpy.case "int({}.to_torch().item())" args %arg1 : (!emitpy.opaque<"ttnn.Tensor">) branches {
  %0 = emitpy.case "int({}.to_torch().item())" args %idx : (!emitpy.opaque<"ttnn.Tensor">) branches {
    %1 = emitpy.call_opaque "ttnn.abs"(%arg0) : (!emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">
    emitpy.case_yield %1 : !emitpy.opaque<"ttnn.Tensor">
  }, {
    emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
  } -> (!emitpy.opaque<"ttnn.Tensor">)
  return %0 : !emitpy.opaque<"ttnn.Tensor">
}

// -----

// Three branches, so the parser has to keep reading comma-separated regions.
// CHECK-LABEL: func.func @three_branch
func.func @three_branch(%arg0: !emitpy.opaque<"ttnn.Tensor">, %idx: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
  // CHECK: emitpy.case
  // CHECK-COUNT-3: emitpy.case_yield
  %0 = emitpy.case "{}" args %idx : (!emitpy.opaque<"ttnn.Tensor">) branches {
    emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
  }, {
    emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
  }, {
    emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
  } -> (!emitpy.opaque<"ttnn.Tensor">)
  return %0 : !emitpy.opaque<"ttnn.Tensor">
}

// -----

// Nothing produced, so the result clause is omitted and the implicit terminator
// is not printed.
// CHECK-LABEL: func.func @nothing_produced
func.func @nothing_produced(%idx: !emitpy.opaque<"ttnn.Tensor">) {
  // CHECK: emitpy.case "{}" args %arg0 : (!emitpy.opaque<"ttnn.Tensor">) branches {
  // CHECK-NOT: ->
  emitpy.case "{}" args %idx : (!emitpy.opaque<"ttnn.Tensor">) branches {
  }
  return
}

// -----

// Two produced values, to pin down the multi-result form.
// CHECK-LABEL: func.func @two_produced
func.func @two_produced(%arg0: !emitpy.opaque<"ttnn.Tensor">, %idx: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
  // CHECK: emitpy.case
  // CHECK: -> (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">)
  %0:2 = emitpy.case "{}" args %idx : (!emitpy.opaque<"ttnn.Tensor">) branches {
    emitpy.case_yield %arg0, %arg0 : !emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">
  }, {
    emitpy.case_yield %arg0, %arg0 : !emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">
  } -> (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">)
  return %0#0 : !emitpy.opaque<"ttnn.Tensor">
}

// -----

// A case nested in a while body.
// CHECK-LABEL: func.func @nested
func.func @nested(%arg0: !emitpy.opaque<"ttnn.Tensor">, %idx: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
  // CHECK: emitpy.while
  // CHECK: emitpy.case
  %0 = emitpy.while trip_count 4 inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
  ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">):
    %1 = emitpy.case "{}" args %idx : (!emitpy.opaque<"ttnn.Tensor">) branches {
      emitpy.case_yield %acc : !emitpy.opaque<"ttnn.Tensor">
    }, {
      emitpy.case_yield %arg0 : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    emitpy.while_yield %1 : !emitpy.opaque<"ttnn.Tensor">
  } -> (!emitpy.opaque<"ttnn.Tensor">)
  return %0 : !emitpy.opaque<"ttnn.Tensor">
}
