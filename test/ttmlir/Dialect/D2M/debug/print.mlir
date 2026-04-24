// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

module {
  // d2m.print with no arguments
  // CHECK-LABEL: func.func private @print_no_args
  func.func private @print_no_args() attributes {d2m.thread = #d2m.thread<compute>} {
    d2m.print("done\n") : () -> ()
    // CHECK: ttkernel.dprint("done\0A") : () -> ()
    return
  }

  // d2m.print with a single integer argument
  // CHECK-LABEL: func.func private @print_one_arg
  func.func private @print_one_arg() attributes {d2m.thread = #d2m.thread<compute>} {
    %x = arith.constant 42 : i32
    d2m.print("x={}\n", %x) : (i32) -> ()
    // CHECK: %[[X:.*]] = arith.constant 42 : i32
    // CHECK: ttkernel.dprint("x={}\0A", %[[X]]) : (i32) -> ()
    return
  }

  // d2m.print with multiple arguments
  // CHECK-LABEL: func.func private @print_multi_args
  func.func private @print_multi_args() attributes {d2m.thread = #d2m.thread<compute>} {
    %x = arith.constant 1 : i32
    %y = arith.constant 2 : i32
    d2m.print("x={} y={}\n", %x, %y) : (i32, i32) -> ()
    // CHECK: ttkernel.dprint("x={} y={}\0A", %{{.*}}, %{{.*}}) : (i32, i32) -> ()
    return
  }
}
