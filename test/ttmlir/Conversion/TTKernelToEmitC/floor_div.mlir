// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: FileCheck %s --input-file=%t

// Arith FloorDivSIOp doesn't have an emitc lowering, probably because of the spec
// which says:
//   Signed integer division. Rounds towards negative infinity, i.e. 5 / -2 = -3
//
// However we know our index type will map to size_t which is unsigned, making a
// negative denominator impossible, so as long as we assert that this floordiv
// is working on values of `index` type it's safe to map this op to regular
// divi.

// CHECK-LABEL: func @floor_div
func.func @floor_div() -> index attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> i32
  %1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> i32
  %2 = arith.index_cast %0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  // CHECK: emitc.div
  %4 = arith.floordivsi %2, %3 : index
  return %4 : index
}
