// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
!slot = memref<1x1x!ttcore.tile<32x32, f32>, #l1>

module {
  // CHECK-LABEL: func.func private @sender
  // CHECK: %[[PORT:.*]] = arith.constant 31 : i32
  // CHECK: %[[CB:.*]] = ttkernel.get_common_arg_val
  // CHECK: ttkernel.remote_cb_reserve_back(%[[PORT]], %{{.*}})
  // CHECK: %[[ADDR:.*]] = ttkernel.get_read_ptr(%[[CB]])
  // CHECK: ttkernel.remote_cb_push_back_and_write_pages(%[[PORT]], %[[ADDR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
  // CHECK: ttkernel.update_remote_cb_config_in_l1(%{{.*}})
  func.func private @sender() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %gcb = d2m.get_arg(0) : !d2m.global_cb<!slot>
    %src = d2m.get_arg(1) : !slot
    d2m.global_cb_reserve %gcb : !d2m.global_cb<!slot> -> !slot
    d2m.global_cb_push %gcb, %src : !d2m.global_cb<!slot>, !slot
    return
  }

  // CHECK-LABEL: func.func private @receiver
  // CHECK: %[[PORT:.*]] = arith.constant 31 : i32
  // CHECK: ttkernel.remote_cb_wait_front(%[[PORT]], %{{.*}})
  // CHECK: ttkernel.remote_cb_pop_front(%[[PORT]], %{{.*}})
  // CHECK: ttkernel.update_remote_cb_config_in_l1(%{{.*}})
  func.func private @receiver() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %gcb = d2m.get_arg(0) : !d2m.global_cb<!slot>
    %got = d2m.global_cb_wait %gcb : !d2m.global_cb<!slot> -> !slot
    d2m.global_cb_pop %gcb : !d2m.global_cb<!slot>
    return
  }
}
