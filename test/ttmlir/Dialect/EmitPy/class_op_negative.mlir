// RUN: ttmlir-opt %s -split-input-file -verify-diagnostics

// Test: At most one __init__
module {
  // expected-error @+1 {{class body must have at most one __init__}}
  emitpy.class @TooManyInits {
    func.func @__init__(%self: !emitpy.opaque<"C">) {
      return
    }
    func.func @__init__(%self: !emitpy.opaque<"C">) {
      return
    }
  }
}

// -----

// Test: __init__ must not return a value
module {
  emitpy.class @InitReturnsValue {
    // expected-error @+1 {{'func.func' op __init__ must not return a value}}
    func.func @__init__(%self: !emitpy.opaque<"C">) -> index {
      %0 = emitpy.literal "0" : index
      return %0 : index
    }
  }
}

// -----

// Test: Invalid emitpy.method_kind
module {
  emitpy.class @BadMethodKind {
    // expected-error @+1 {{emitpy.method_kind must be one of 'instance', 'staticmethod', or 'classmethod'}}
    func.func @f(%self: !emitpy.opaque<"C">) attributes {emitpy.method_kind = "bogus"} {
      return
    }
  }
}

// -----

// Test: Instance method must have a receiver argument
module {
  emitpy.class @MissingReceiver {
    // expected-error @+1 {{instance and class methods must take a receiver argument}}
    func.func @forward() {
      return
    }
  }
}

// -----

// Test: Class method receiver must be named cls when provided
module {
  emitpy.class @ClassMethodBadArgName {
    // expected-error @+1 {{first argument must be named 'cls' via emitpy.name}}
    func.func @make(%self: !emitpy.opaque<"C"> {emitpy.name = "self"}) attributes {emitpy.method_kind = "classmethod"} {
      return
    }
  }
}

// -----

// Test: Only EmitPy or func ops are allowed in class body
module {
  emitpy.class @DisallowNonEmitPyOp {
    // expected-error @+1 {{only emitpy or func operations are allowed in a class body}}
    arith.constant 0 : i32
  }
}
