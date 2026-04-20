// RUN: ttmlir-translate --mlir-to-python %s | FileCheck %s

// This test validates that FileOp is correctly translated to Python.
// Multiple emitpy.file ops should generate their content sequentially.

module {
  // CHECK-LABEL: # File: "main"
  emitpy.file "main" {
    // CHECK: import ttnn
    emitpy.import import "ttnn"
    // CHECK: import utils
    emitpy.import import "utils"

    // CHECK: def add(input):
    func.func @add(%arg0: !emitpy.opaque<"[ttnn.Tensor]"> {emitpy.name = "input"}) -> !emitpy.opaque<"[ttnn.Tensor]"> {
      %0 = emitpy.expression (%arg0) : (!emitpy.opaque<"[ttnn.Tensor]">) -> !emitpy.opaque<"ttnn.Tensor"> {
      ^bb0(%arg1: !emitpy.opaque<"[ttnn.Tensor]">):
        %idx = emitpy.literal "0" : index
        %tensor = emitpy.subscript %arg1[%idx] : (!emitpy.opaque<"[ttnn.Tensor]">, index) -> !emitpy.opaque<"ttnn.Tensor">
        emitpy.yield %tensor : !emitpy.opaque<"ttnn.Tensor">
      }
      %1 = emitpy.call_opaque "ttnn.add"(%0, %0) : (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">
      %2 = emitpy.call_opaque "util_create_list"(%1) : (!emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"[ttnn.Tensor]">
      return %2 : !emitpy.opaque<"[ttnn.Tensor]">
    }

    // CHECK: def main():
    func.func @main() -> i32 {
      %input = emitpy.call_opaque "create_input"() : () -> !emitpy.opaque<"[ttnn.Tensor]">
      %result = call @add(%input) : (!emitpy.opaque<"[ttnn.Tensor]">) -> !emitpy.opaque<"[ttnn.Tensor]">
      %c0 = "emitpy.constant"() <{value = 0 : i32}> : () -> i32
      return %c0 : i32
    }

    // CHECK-NOT: def get_device():
  }

  // CHECK-LABEL: # File: "utils"
  emitpy.file "utils" {
    // CHECK: import ttnn
    emitpy.import import "ttnn"

    // CHECK: def get_device():
    func.func @get_device() -> !emitpy.opaque<"ttnn.Device"> {
      %0 = emitpy.call_opaque "ttnn.open_device"() : () -> !emitpy.opaque<"ttnn.Device">
      return %0 : !emitpy.opaque<"ttnn.Device">
    }
  }
}
