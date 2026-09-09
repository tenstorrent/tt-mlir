// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: FileCheck %s --input-file=%t

// Arith MinSIOp / MaxSIOp don't have a built-in emitc lowering. The
// TTKernelToEmitC pass rewrites them to std::min<int32_t> / std::max<int32_t>
// CallOpaqueOps, mirroring the unsigned variants.

// CHECK-LABEL: func @minsi
func.func @minsi() -> i32 attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> i32
  %1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> i32
  // CHECK: emitc.call_opaque "std::min<int32_t>"
  %2 = arith.minsi %0, %1 : i32
  return %2 : i32
}

// CHECK-LABEL: func @maxsi
func.func @maxsi() -> i32 attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> i32
  %1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> i32
  // CHECK: emitc.call_opaque "std::max<int32_t>"
  %2 = arith.maxsi %0, %1 : i32
  return %2 : i32
}
