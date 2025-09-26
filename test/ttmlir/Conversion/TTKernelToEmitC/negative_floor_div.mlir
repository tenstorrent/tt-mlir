// RUN: not ttmlir-opt --convert-ttkernel-to-emitc %s 2>&1 | FileCheck %s

// Arith FloorDivSIOp doesn't have an emitc lowering, probably because of the spec
// which says:
//   Signed integer division. Rounds towards negative infinity, i.e. 5 / -2 = -3
//
// However we know our index type will map to size_t which is unsigned, making a
// negative denominator impossible, so as long as we assert that this floordiv
// is working on values of `index` type it's safe to map this op to regular
// divi.

// CHECK: error: failed to legalize operation 'arith.floordivsi'

func.func @negative_floor_div() -> i32 attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> i32
  %1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> i32
  %2 = arith.floordivsi %0, %1 : i32
  return %2 : i32
}
