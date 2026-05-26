// RUN: ttmlir-opt --convert-ttnn-to-emitpy %s | FileCheck %s

module {
  func.func @test_global_semaphore() {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: emitpy.call_opaque "ttnn.create_global_semaphore"
    // CHECK-SAME: keyword_args = ["mesh_device", "cores", "initial_value"]
    // CHECK-SAME: !emitpy.opaque<"ttnn.GlobalSemaphore">
    %sem = "ttnn.create_global_semaphore"(%0) <{core_range_set = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,7)>]>, initial_value = 0 : ui32}> : (!ttnn.device) -> !ttnn.global_semaphore
    // CHECK: emitpy.call_opaque "ttnn.reset_global_semaphore_value"
    // CHECK-SAME: keyword_args = ["", "value"]
    "ttnn.reset_global_semaphore"(%sem) <{value = 0 : ui32}> : (!ttnn.global_semaphore) -> ()
    return
  }
}
