// RUN: ttmlir-opt %s -split-input-file | FileCheck %s

// Test: FileOp with single function
module {
  // CHECK: emitpy.file "main"
  emitpy.file "main" {
    // CHECK: func.func @simple_func
    func.func @simple_func(%arg0: !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor"> {
      return %arg0 : !emitpy.opaque<"ttnn.Tensor">
    }
  }
}

// -----

// Test: FileOp with multiple functions
module {
  // CHECK: emitpy.file "utils"
  emitpy.file "utils" {
    // CHECK: func.func @func_one
    func.func @func_one() -> index {
      %0 = emitpy.literal "0" : index
      return %0 : index
    }

    // CHECK: func.func @func_two
    func.func @func_two(%arg0: !emitpy.opaque<"float">) -> !emitpy.opaque<"float"> {
      %0 = emitpy.call_opaque "relu"(%arg0) : (!emitpy.opaque<"float">) -> !emitpy.opaque<"float">
      return %0 : !emitpy.opaque<"float">
    }
  }
}

// -----

// Test: Multiple FileOps in same module with cross-file function calls
module {
  // CHECK: emitpy.file "main"
  emitpy.file "main" {
    // CHECK: emitpy.import import "ttnn"
    emitpy.import import "ttnn"
    // CHECK: emitpy.import from "consteval" import "execute_0"
    emitpy.import from "consteval" import "execute_0"

    // CHECK: func.func @forward
    func.func @forward(%arg0: !emitpy.opaque<"ttnn.Tensor">, %arg1: !emitpy.opaque<"ttnn.Device">) -> !emitpy.opaque<"ttnn.Tensor"> {
      // Call consteval function to get precomputed weight
      %weight = emitpy.call_opaque "execute_0"(%arg1) : (!emitpy.opaque<"ttnn.Device">) -> !emitpy.opaque<"ttnn.Tensor">
      // Use weight in computation
      %0 = emitpy.call_opaque "ttnn.add"(%arg0, %weight) : (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">
      return %0 : !emitpy.opaque<"ttnn.Tensor">
    }
  }

  // CHECK: emitpy.file "consteval"
  emitpy.file "consteval" {
    // CHECK: emitpy.import import "ttnn"
    emitpy.import import "ttnn"

    // CHECK: func.func @execute_0
    func.func @execute_0(%arg0: !emitpy.opaque<"ttnn.Device">) -> !emitpy.opaque<"ttnn.Tensor"> {
      // Consteval function that creates a constant weight tensor
      %weight = emitpy.call_opaque "ttnn.zeros"(%arg0) : (!emitpy.opaque<"ttnn.Device">) -> !emitpy.opaque<"ttnn.Tensor">
      return %weight : !emitpy.opaque<"ttnn.Tensor">
    }
  }
}

// -----

// Test: FileOp with globals
module {
  // CHECK: emitpy.file "main"
  emitpy.file "main" {
    // CHECK: emitpy.global @device = #emitpy.opaque<"None">
    emitpy.global @device = #emitpy.opaque<"None">

    // CHECK: func.func @init_device
    func.func @init_device() -> !emitpy.opaque<"ttnn.Device"> {
      %0 = emitpy.call_opaque "ttnn.open_device"() : () -> !emitpy.opaque<"ttnn.Device">
      emitpy.assign_global @device = %0 : !emitpy.opaque<"ttnn.Device">
      return %0 : !emitpy.opaque<"ttnn.Device">
    }
  }
}

// -----

// Test: FileOp with numeric characters and underscore in id
module {
  // CHECK: emitpy.file "file_123"
  emitpy.file "file_123" {
    func.func @test() {
      return
    }
  }
}

// -----

// Test: FileOp with nested operations
module {
  // CHECK: emitpy.file "nested"
  emitpy.file "nested" {
    // CHECK: func.func @complex_func
    func.func @complex_func(%arg0: index, %arg1: index) -> index {
      %0 = emitpy.expression(%arg0, %arg1) : (index, index) -> index {
      ^bb0(%a: index, %b: index):
        %1 = emitpy.call_opaque "add"(%a, %b) : (index, index) -> index
        emitpy.yield %1 : index
      }
      return %0 : index
    }
  }
}

// -----

// Test: FileOp with mixed content
module {
  // CHECK: emitpy.file "mixed"
  emitpy.file "mixed" {
    // CHECK: emitpy.import import "ttnn"
    emitpy.import import "ttnn"

    // CHECK: emitpy.global @counter = 0 : i64
    emitpy.global @counter = 0 : i64

    // CHECK: func.func @increment
    func.func @increment() -> i64 {
      %0 = emitpy.global_statement @counter : i64
      %1 = "emitpy.constant"() <{value = 1 : i64}> : () -> i64
      %2 = emitpy.call_opaque "add"(%0, %1) : (i64, i64) -> i64
      emitpy.assign_global @counter = %2 : i64
      return %2 : i64
    }
  }
}

// -----

// Test: FileOp with SymbolTable trait - fileOp can have SymbolRefAttr inside its region
module {
  // CHECK: emitpy.file "symbols"
  emitpy.file "symbols" {
    // CHECK: func.func @symbol_func
    func.func @symbol_func() {
      return
    }
  }
}

// -----

// Test: FileOp with IsolatedFromAbove trait
module {
  // CHECK: emitpy.file "isolated"
  emitpy.file "isolated" {
    // CHECK: func.func @isolated_func
    func.func @isolated_func() -> i64 {
      %c = arith.constant 42 : i64
      return %c : i64
    }
  }
}
