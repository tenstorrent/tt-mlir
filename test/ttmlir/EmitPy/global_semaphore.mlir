// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --convert-ttnn-to-emitpy %s | FileCheck %s

module {
  func.func @test_global_semaphore() {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: emitpy.call_opaque "ttnn.create_global_semaphore"
    %sem = "ttnn.create_global_semaphore"() <{initial_value = 0 : ui32, core_range = #ttnn.core_range<(0,0), (7,7)>}> : () -> !ttnn.global_semaphore
    // CHECK: emitpy.call_opaque "ttnn.reset_global_semaphore_value"
    "ttnn.reset_global_semaphore"(%sem) <{value = 0 : ui32}> : (!ttnn.global_semaphore) -> ()
    return
  }
}
