// RUN: ttmlir-opt --split-input-file %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// Test EmitPy while op.

module {
  func.func @counted(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK-LABEL: func.func @counted
    // CHECK: emitpy.while trip_count 4 inits(%{{.*}} : !emitpy.opaque<"ttnn.Tensor">) {
    // CHECK:   emitpy.while_yield
    // CHECK: } -> (!emitpy.opaque<"ttnn.Tensor">)
    %0 = emitpy.while trip_count 4 inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">):
      %1 = emitpy.call_opaque "ttnn.add"(%acc, %acc) : (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">
      emitpy.while_yield %1 : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  func.func @condition_with_args(%arg0: !emitpy.opaque<"ttnn.Tensor">,
                                 %arg1: !emitpy.opaque<"int">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK-LABEL: func.func @condition_with_args
    // CHECK: emitpy.while "{} > 0" args %{{.*}} : (!emitpy.opaque<"int">) inits(%{{.*}} : !emitpy.opaque<"ttnn.Tensor">) {
    %0 = emitpy.while "{} > 0" args %arg1 : (!emitpy.opaque<"int">)
        inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">):
      %1 = emitpy.call_opaque "ttnn.add"(%acc, %acc) : (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">
      emitpy.while_yield %1 : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  // A loop that carries nothing: the body runs for its side effects only.
  // With nothing to carry the terminator is implicit, so it is not printed.
  func.func @nothing_carried() -> () {
    // CHECK-LABEL: func.func @nothing_carried
    // CHECK: emitpy.while "True" {
    // CHECK-NEXT: emitpy.call_opaque "step"
    // CHECK-NEXT: }
    emitpy.while "True" {
      emitpy.call_opaque "step"() : () -> ()
      emitpy.while_yield
    }
    return
  }
}

// -----

module {
  // Two carried values swapped by the body. Nothing here needs the emitter,
  // but it is the shape the tuple-assignment carry-back exists for.
  func.func @swap(%arg0: !emitpy.opaque<"ttnn.Tensor">,
                  %arg1: !emitpy.opaque<"ttnn.Tensor">)
      -> (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) {
    // CHECK-LABEL: func.func @swap
    // CHECK: emitpy.while trip_count 2 inits(%{{.*}}, %{{.*}} : !emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) {
    // CHECK:   emitpy.while_yield %{{.*}}, %{{.*}} : !emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">
    %0:2 = emitpy.while trip_count 2
        inits(%arg0, %arg1 : !emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%a: !emitpy.opaque<"ttnn.Tensor">, %b: !emitpy.opaque<"ttnn.Tensor">):
      emitpy.while_yield %b, %a : !emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">)
    return %0#0, %0#1 : !emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">
  }
}

// -----

module {
  // A while nested in a while body.
  func.func @nested(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK-LABEL: func.func @nested
    // CHECK: emitpy.while trip_count 2
    // CHECK:   emitpy.while trip_count 3
    %0 = emitpy.while trip_count 2 inits(%arg0 : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%outer: !emitpy.opaque<"ttnn.Tensor">):
      %1 = emitpy.while trip_count 3 inits(%outer : !emitpy.opaque<"ttnn.Tensor">) {
      ^bb0(%inner: !emitpy.opaque<"ttnn.Tensor">):
        %2 = emitpy.call_opaque "ttnn.add"(%inner, %inner) : (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">
        emitpy.while_yield %2 : !emitpy.opaque<"ttnn.Tensor">
      } -> (!emitpy.opaque<"ttnn.Tensor">)
      emitpy.while_yield %1 : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}
