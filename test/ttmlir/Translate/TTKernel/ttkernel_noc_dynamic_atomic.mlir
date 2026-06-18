// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: ttmlir-translate --ttkernel-to-cpp -o %t.cpp %t
// RUN: FileCheck %s --input-file=%t.cpp

// CHECK: #include "api/dataflow/dataflow_api.h"
// CHECK: #include "api/dataflow/noc.h"
// CHECK: void kernel_main
func.func @ttkernel_dynamic_noc_atomic_barrier() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  // CHECK: int32_t [[NOC_ARG:.*]] = 262400
  %noc_arg = arith.constant 262400 : i32
  %noc_ptr = ttkernel.reinterpret_cast(%noc_arg) : (i32) -> (!ttkernel.l1_addr_ptr<8>)
  %noc_offset = arith.constant 0 : i32
  // CHECK: int8_t [[NOC:v[0-9]+]] = (int8_t)
  %noc = ttkernel.load_from_l1(%noc_ptr, %noc_offset) : (!ttkernel.l1_addr_ptr<8>, i32) -> i8
  // CHECK: Noc([[NOC]]).async_atomic_barrier()
  ttkernel.noc_async_atomic_barrier(%noc) : (i8) -> ()
  // CHECK: return
  func.return
}
