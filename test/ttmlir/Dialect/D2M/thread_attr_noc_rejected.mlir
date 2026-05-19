// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s

module {
  // CHECK: error: expected attribute value
  func.func private @noc_is_not_thread_placement() attributes {
    d2m.thread = #d2m.thread<datamovement, noc = 0>
  } {
    return
  }
}
