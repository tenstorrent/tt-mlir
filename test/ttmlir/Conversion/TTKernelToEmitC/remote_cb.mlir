// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @remote_cb_sender
  // CHECK: emitc.call_opaque "experimental::remote_cb_reserve_back"
  // CHECK: emitc.call_opaque "experimental::remote_cb_push_back_and_write_pages"
  // CHECK: emitc.call_opaque "experimental::update_remote_cb_config_in_l1"
  func.func @remote_cb_sender() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %port = arith.constant 31 : i32
    %addr = arith.constant 4096 : i32
    %pages = arith.constant 1 : i32
    %rows = arith.constant 1 : i32
    %coal = arith.constant 1 : i32
    %psz = arith.constant 4096 : i32
    "ttkernel.experimental.remote_cb_reserve_back"(%port, %pages) : (i32, i32) -> ()
    "ttkernel.experimental.remote_cb_push_back_and_write_pages"(%port, %addr, %pages, %rows, %coal, %psz) : (i32, i32, i32, i32, i32, i32) -> ()
    "ttkernel.experimental.update_remote_cb_config_in_l1"(%port) : (i32) -> ()
    return
  }

  // CHECK-LABEL: func.func @remote_cb_receiver
  // CHECK: emitc.call_opaque "experimental::remote_cb_wait_front"
  // CHECK: emitc.call_opaque "experimental::remote_cb_pop_front"
  func.func @remote_cb_receiver() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %port = arith.constant 31 : i32
    %pages = arith.constant 1 : i32
    "ttkernel.experimental.remote_cb_wait_front"(%port, %pages) : (i32, i32) -> ()
    "ttkernel.experimental.remote_cb_pop_front"(%port, %pages) : (i32, i32) -> ()
    return
  }
}
