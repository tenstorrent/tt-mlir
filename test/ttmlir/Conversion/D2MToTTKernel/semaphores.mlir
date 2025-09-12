// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

module {
  // Local semaphore_set
  func.func private @local_set(%sem0: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<datamovement>} {
    %c1 = arith.constant 1 : index
    d2m.semaphore_set %sem0, %c1
    // CHECK: %[[CTARG:[0-9]+]] = ttkernel.get_compile_time_arg_val(0) : () -> i32
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore(%[[CTARG]])
    // CHECK: %[[PTR:[0-9]+]] = ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>(%[[SEM]])
    // CHECK: ttkernel.noc_semaphore_set(%[[PTR]], %c1)
    return
  }

  // Remote single-core semaphore_inc
  func.func private @remote_inc(%sem0: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<datamovement>} {
    %c1 = arith.constant 1 : index
    %y  = arith.constant 2 : index
    %x  = arith.constant 3 : index
    d2m.semaphore_inc %sem0, %c1, core[%y, %x]
    // CHECK: %[[CTARG:[0-9]+]] = ttkernel.get_compile_time_arg_val(0) : () -> i32
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore(%[[CTARG]])
    // CHECK: %[[NOC:[0-9]+]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[SEM]])
    // CHECK: ttkernel.noc_semaphore_inc(%[[NOC]], %c1)
    return
  }

  // Remote multicast semaphore_set
  func.func private @mcast_set(%sem0: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<datamovement>} {
    %c7 = arith.constant 7 : index
    %y  = arith.constant 4 : index
    %x  = arith.constant 5 : index
    %h  = arith.constant 2 : index
    %w  = arith.constant 3 : index
    d2m.semaphore_set %sem0, %c7, core[%y, %x] mcast[%h, %w]
    // CHECK: %[[CTARG:[0-9]+]] = ttkernel.get_compile_time_arg_val(0) : () -> i32
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore(%[[CTARG]])
    // CHECK: %[[MADDR:[0-9]+]] = ttkernel.experimental::get_noc_multicast_addr({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[SEM]])
    // CHECK: %[[PTR:[0-9]+]] = ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>(%[[SEM]])
    // CHECK: ttkernel.noc_semaphore_set(%[[PTR]], %c7)
    // CHECK: ttkernel.noc_semaphore_set_multicast(%[[SEM]], %[[MADDR]], {{.*}})
    return
  }

  // semaphore_wait without reset
  func.func private @wait_no_reset(%sem0: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<datamovement>} {
    %c2 = arith.constant 2 : index
    d2m.semaphore_wait %sem0, %c2
    // CHECK: %[[CTARG:[0-9]+]] = ttkernel.get_compile_time_arg_val(0) : () -> i32
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore(%[[CTARG]])
    // CHECK: %[[PTR:[0-9]+]] = ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>(%[[SEM]])
    // CHECK: ttkernel.noc_semaphore_wait(%[[PTR]], %c2)
    return
  }

  // semaphore_wait with reset
  func.func private @wait_with_reset(%sem0: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<datamovement>} {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    d2m.semaphore_wait %sem0, %c2 reset %c0
    // CHECK: %[[CTARG:[0-9]+]] = ttkernel.get_compile_time_arg_val(0) : () -> i32
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore(%[[CTARG]])
    // CHECK: %[[PTR:[0-9]+]] = ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>(%[[SEM]])
    // CHECK: ttkernel.noc_semaphore_wait(%[[PTR]], %c2)
    // CHECK: ttkernel.noc_semaphore_set(%[[PTR]], %c0)
    return
  }

}
