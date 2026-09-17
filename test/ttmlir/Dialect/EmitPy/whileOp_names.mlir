// RUN: ttmlir-translate --mlir-to-python -o %t.py %s
// RUN: FileCheck %s --input-file=%t.py
// The loop-carried locals are named through the emitter's name table, so a
// name the IR already claimed is renamed rather than silently shared.

module {
  func.func @collide(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
    // CHECK: carried_0 = ttnn.abs(
    %pre = emitpy.call_opaque "ttnn.abs"(%arg0) {emitpy.name = "carried_0"} : (!emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">

    // CHECK: carried_0_{{[0-9]+}} = carried_0
    // CHECK: for _ in range(2):
    %0 = emitpy.while trip_count 2 inits(%pre : !emitpy.opaque<"ttnn.Tensor">) {
    ^bb0(%acc: !emitpy.opaque<"ttnn.Tensor">):
      // CHECK: ttnn.add(carried_0_{{[0-9]+}}, carried_0_{{[0-9]+}})
      %1 = emitpy.call_opaque "ttnn.add"(%acc, %acc) : (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">
      emitpy.while_yield %1 : !emitpy.opaque<"ttnn.Tensor">
    } -> (!emitpy.opaque<"ttnn.Tensor">)

    // The result reads the same local the body assigned.
    // CHECK: carried_0_{{[0-9]+}} = var
    // CHECK: return carried_0_{{[0-9]+}}
    return %0 : !emitpy.opaque<"ttnn.Tensor">
  }
}
