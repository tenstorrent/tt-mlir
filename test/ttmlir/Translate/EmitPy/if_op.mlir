// RUN: ttmlir-translate -mlir-to-python %s | FileCheck %s
// Test EmitPy if op Python translation.

module {
  emitpy.global @_cached = #emitpy.opaque<"None">

  // CHECK-LABEL: def test_if_with_cache
  func.func @test_if_with_cache(%arg0: !emitpy.opaque<"dict"> {emitpy.name = "inputs"}) -> !emitpy.opaque<"dict"> {
    %0 = emitpy.global_statement @_cached : !emitpy.opaque<"dict">
    // CHECK: global _cached
    // CHECK: if _cached is None:
    // CHECK:   _cached = inputs
    emitpy.if "{} is None" args %0 : (!emitpy.opaque<"dict">) {
      emitpy.assign_global @_cached = %arg0 : !emitpy.opaque<"dict">
    }
    return %0 : !emitpy.opaque<"dict">
  }

  // CHECK-LABEL: def test_if_else
  func.func @test_if_else(%arg0: !emitpy.opaque<"dict"> {emitpy.name = "d"}) -> () {
    // CHECK: if d is None:
    // CHECK:   compute(d)
    // CHECK: else:
    // CHECK:   fallback(d)
    emitpy.if "{} is None" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.call_opaque "compute"(%arg0) : (!emitpy.opaque<"dict">) -> ()
    } else {
      emitpy.call_opaque "fallback"(%arg0) : (!emitpy.opaque<"dict">) -> ()
    }
    return
  }

  // CHECK-LABEL: def test_elif
  func.func @test_elif(%arg0: !emitpy.opaque<"dict"> {emitpy.name = "d"},
                        %arg1: !emitpy.opaque<"int"> {emitpy.name = "n"}) -> () {
    // CHECK: if d is None:
    // CHECK:   compute(d)
    // CHECK: elif n > 0:
    // CHECK:   refresh(n)
    emitpy.if "{} is None" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.call_opaque "compute"(%arg0) : (!emitpy.opaque<"dict">) -> ()
    } else {
      emitpy.if "{} > 0" args %arg1 : (!emitpy.opaque<"int">) {
        emitpy.call_opaque "refresh"(%arg1) : (!emitpy.opaque<"int">) -> ()
      }
    }
    return
  }

  // CHECK-LABEL: def test_else_with_nested_if_else_not_elif
  func.func @test_else_with_nested_if_else_not_elif(%arg0: !emitpy.opaque<"dict"> {emitpy.name = "d"},
                                                     %arg1: !emitpy.opaque<"int"> {emitpy.name = "n"}) -> () {
    // CHECK: if d is None:
    // CHECK:   compute(d)
    // CHECK: else:
    // CHECK:   if n > 0:
    // CHECK:     refresh(n)
    // CHECK:   else:
    // CHECK:     fallback(d)
    emitpy.if "{} is None" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.call_opaque "compute"(%arg0) : (!emitpy.opaque<"dict">) -> ()
    } else {
      emitpy.if "{} > 0" args %arg1 : (!emitpy.opaque<"int">) {
        emitpy.call_opaque "refresh"(%arg1) : (!emitpy.opaque<"int">) -> ()
      } else {
        emitpy.call_opaque "fallback"(%arg0) : (!emitpy.opaque<"dict">) -> ()
      }
    }
    return
  }

  // CHECK-LABEL: def test_else_with_nested_if_not_elif
  func.func @test_else_with_nested_if_not_elif(%arg0: !emitpy.opaque<"dict"> {emitpy.name = "d"},
                                                %arg1: !emitpy.opaque<"int"> {emitpy.name = "n"}) -> () {
    // CHECK: if d is None:
    // CHECK:   compute(d)
    // CHECK: else:
    // CHECK:   if n > 0:
    // CHECK:     refresh(n)
    // CHECK:   foo()
    emitpy.if "{} is None" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.call_opaque "compute"(%arg0) : (!emitpy.opaque<"dict">) -> ()
    } else {
      emitpy.if "{} > 0" args %arg1 : (!emitpy.opaque<"int">) {
        emitpy.call_opaque "refresh"(%arg1) : (!emitpy.opaque<"int">) -> ()
      }
      emitpy.call_opaque "foo"() : () -> ()
    }
    return
  }

  // CHECK-LABEL: def test_nested_else_with_elif
  func.func @test_nested_else_with_elif(%arg0: !emitpy.opaque<"dict"> {emitpy.name = "d"},
                                         %arg1: !emitpy.opaque<"int"> {emitpy.name = "n"},
                                         %arg2: !emitpy.opaque<"int"> {emitpy.name = "m"}) -> () {
    // CHECK: if d is None:
    // CHECK:   compute(d)
    // CHECK: else:
    // CHECK:   if n > 0:
    // CHECK:     refresh(n)
    // CHECK:   elif m > 0:
    // CHECK:     refresh(m)
    emitpy.if "{} is None" args %arg0 : (!emitpy.opaque<"dict">) {
      emitpy.call_opaque "compute"(%arg0) : (!emitpy.opaque<"dict">) -> ()
    } else {
      emitpy.if "{} > 0" args %arg1 : (!emitpy.opaque<"int">) {
        emitpy.call_opaque "refresh"(%arg1) : (!emitpy.opaque<"int">) -> ()
      } else {
        emitpy.if "{} > 0" args %arg2 : (!emitpy.opaque<"int">) {
          emitpy.call_opaque "refresh"(%arg2) : (!emitpy.opaque<"int">) -> ()
        }
      }
    }
    return
  }

  // CHECK-LABEL: def test_if_no_args
  func.func @test_if_no_args() -> () {
    // CHECK: if True:
    // CHECK:   do_something()
    emitpy.if "True" {
      emitpy.call_opaque "do_something"() : () -> ()
    }
    return
  }

  // CHECK-LABEL: def test_if_no_args_else
  func.func @test_if_no_args_else() -> () {
    // CHECK: if True:
    // CHECK:   branch_a()
    // CHECK: else:
    // CHECK:   branch_b()
    emitpy.if "True" {
      emitpy.call_opaque "branch_a"() : () -> ()
    } else {
      emitpy.call_opaque "branch_b"() : () -> ()
    }
    return
  }
}
