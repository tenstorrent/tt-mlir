// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s

// Verify all valid forms of #d2m.thread<...> round-trip.
// Historically, the assembly format had two optional groups both led by a
// comma, which caused `#d2m.thread<datamovement, noc = 0>` to be misparsed
// as if the kernel symbol position contained `noc = 0`.

module {
  // CHECK: func.func private @no_kernel_no_noc() attributes {d2m.thread = #d2m.thread<compute>}
  func.func private @no_kernel_no_noc() attributes {d2m.thread = #d2m.thread<compute>} {
    return
  }

  // CHECK: func.func private @kernel_only() attributes {d2m.thread = #d2m.thread<datamovement, @some_sym>}
  func.func private @kernel_only() attributes {d2m.thread = #d2m.thread<datamovement, @some_sym>} {
    return
  }

  // The case from the bug report: noc only, no kernel symbol.
  // CHECK: func.func private @noc_only_0() attributes {d2m.thread = #d2m.thread<datamovement, noc = 0>}
  func.func private @noc_only_0() attributes {d2m.thread = #d2m.thread<datamovement, noc = 0>} {
    return
  }

  // CHECK: func.func private @noc_only_1() attributes {d2m.thread = #d2m.thread<datamovement, noc = 1>}
  func.func private @noc_only_1() attributes {d2m.thread = #d2m.thread<datamovement, noc = 1>} {
    return
  }

  // CHECK: func.func private @kernel_and_noc() attributes {d2m.thread = #d2m.thread<datamovement, @some_sym, noc = 1>}
  func.func private @kernel_and_noc() attributes {d2m.thread = #d2m.thread<datamovement, @some_sym, noc = 1>} {
    return
  }
}
