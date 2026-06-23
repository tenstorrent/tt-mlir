// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

module {
  // Local semaphore_set
  func.func private @local_set() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %sem0 = d2m.get_arg(0) : !d2m.local_semaphore
    %c1 = arith.constant 1 : index
    d2m.semaphore_set %sem0, %c1 : !d2m.local_semaphore
    // CHECK: %[[ARG:[0-9]+]] = ttkernel.get_common_arg_val
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore(%[[ARG]])
    // CHECK: %[[PTR:[0-9]+]] = ttkernel.reinterpret_cast(%[[SEM]])
    // CHECK: ttkernel.noc_semaphore_set(%[[PTR]], %c1)
    return
  }

  // Remote single-core semaphore_inc
  func.func private @remote_inc() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %sem0 = d2m.get_arg(0) : !d2m.local_semaphore
    %c1 = arith.constant 1 : index
    %y  = arith.constant 2 : index
    %x  = arith.constant 3 : index
    d2m.semaphore_inc %sem0, %c1, core[%y, %x] : !d2m.local_semaphore
    // CHECK: %[[ARG:[0-9]+]] = ttkernel.get_common_arg_val
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore(%[[ARG]])
    // CHECK: %[[NOC_ID:[a-zA-Z0-9_]+]] = arith.constant 1 : i8
    // CHECK: %[[NOC:[0-9]+]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[SEM]], %[[NOC_ID]])
    // CHECK: ttkernel.noc_semaphore_inc(%[[NOC]], %c1, %[[NOC_ID]]) :
    // CHECK-NOT: posted
    return
  }

  // Remote multicast semaphore_set
  func.func private @mcast_set() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 1>} {
    %sem0 = d2m.get_arg(0) : !d2m.local_semaphore
    %c7 = arith.constant 7 : index
    %y  = arith.constant 4 : index
    %x  = arith.constant 5 : index
    %h  = arith.constant 2 : index
    %w  = arith.constant 3 : index
    d2m.semaphore_set %sem0, %c7, core[%y, %x] mcast[%h, %w] : !d2m.local_semaphore
    // CHECK: %[[ARG:[0-9]+]] = ttkernel.get_common_arg_val
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore(%[[ARG]])
    // CHECK: %[[NOC:[a-zA-Z0-9_]+]] = arith.constant 0 : i8
    // CHECK: %[[MADDR:[0-9]+]] = ttkernel.get_noc_multicast_addr({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[SEM]], %[[NOC]])
    // CHECK: %[[PTR:[0-9]+]] = ttkernel.reinterpret_cast(%[[SEM]])
    // CHECK: ttkernel.noc_semaphore_set(%[[PTR]], %c7)
    // CHECK: ttkernel.noc_semaphore_set_multicast(%[[SEM]], %[[MADDR]], {{.*}})
    return
  }

  // Remote multicast semaphore_set on NOC1 flips the start and end coordinate
  // operands before creating the standard multicast address.
  func.func private @mcast_set_noc1() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 0>} {
    %sem0 = d2m.get_arg(0) : !d2m.local_semaphore
    %c7 = arith.constant 7 : index
    %y  = arith.constant 4 : index
    %x  = arith.constant 5 : index
    %h  = arith.constant 2 : index
    %w  = arith.constant 3 : index
    d2m.semaphore_set %sem0, %c7, core[%y, %x] mcast[%h, %w] : !d2m.local_semaphore
    // CHECK-LABEL: func.func private @mcast_set_noc1
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore
    // CHECK: %[[START_Y:[0-9]+]] = ttkernel.experimental.convert_logical_y_to_translated
    // CHECK: %[[START_X:[0-9]+]] = ttkernel.experimental.convert_logical_x_to_translated
    // CHECK: %[[END_Y:[0-9]+]] = ttkernel.experimental.convert_logical_y_to_translated
    // CHECK: %[[END_X:[0-9]+]] = ttkernel.experimental.convert_logical_x_to_translated
    // CHECK: %[[NOC:[a-zA-Z0-9_]+]] = arith.constant 1 : i8
    // CHECK: ttkernel.get_noc_multicast_addr(%[[END_X]], %[[END_Y]], %[[START_X]], %[[START_Y]], %[[SEM]], %[[NOC]])
    return
  }

  // semaphore_wait without reset
  func.func private @wait_no_reset() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %sem0 = d2m.get_arg(0) : !d2m.local_semaphore
    %c2 = arith.constant 2 : index
    d2m.semaphore_wait %sem0, %c2 : !d2m.local_semaphore
    // CHECK: %[[ARG:[0-9]+]] = ttkernel.get_common_arg_val
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore(%[[ARG]])
    // CHECK: %[[PTR:[0-9]+]] = ttkernel.reinterpret_cast(%[[SEM]])
    // CHECK: ttkernel.experimental.semaphore_wait(%[[PTR]], %c2)
    return
  }

  // semaphore_wait with reset
  func.func private @wait_with_reset() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %sem0 = d2m.get_arg(0) : !d2m.local_semaphore
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    d2m.semaphore_wait %sem0, %c2 reset %c0 : !d2m.local_semaphore
    // CHECK: %[[ARG:[0-9]+]] = ttkernel.get_common_arg_val
    // CHECK: %[[SEM:[0-9]+]] = ttkernel.get_semaphore(%[[ARG]])
    // CHECK: %[[PTR:[0-9]+]] = ttkernel.reinterpret_cast(%[[SEM]])
    // CHECK: ttkernel.experimental.semaphore_wait(%[[PTR]], %c2)
    // CHECK: ttkernel.noc_semaphore_set(%[[PTR]], {{%c0(_0)?}})
    return
  }

}
