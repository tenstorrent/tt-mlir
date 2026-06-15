// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s

// Verify all valid forms of #d2m.thread<...> round-trip.
// Datamovement placement is represented by DM core index. NoC selection is
// derived by lowerings that know the target architecture.

module {
  // CHECK: func.func private @no_kernel_no_dm_core() attributes {d2m.thread = #d2m.thread<compute>}
  func.func private @no_kernel_no_dm_core() attributes {d2m.thread = #d2m.thread<compute>} {
    return
  }

  // CHECK: func.func private @kernel_only() attributes {d2m.thread = #d2m.thread<datamovement, @some_sym>}
  func.func private @kernel_only() attributes {d2m.thread = #d2m.thread<datamovement, @some_sym>} {
    return
  }

  // CHECK: func.func private @dm_core_only_0() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 0>}
  func.func private @dm_core_only_0() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 0>} {
    return
  }

  // CHECK: func.func private @dm_core_only_1() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 1>}
  func.func private @dm_core_only_1() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 1>} {
    return
  }

  // CHECK: func.func private @kernel_and_dm_core() attributes {d2m.thread = #d2m.thread<datamovement, @some_sym, dm_core = 1>}
  func.func private @kernel_and_dm_core() attributes {d2m.thread = #d2m.thread<datamovement, @some_sym, dm_core = 1>} {
    return
  }
}
