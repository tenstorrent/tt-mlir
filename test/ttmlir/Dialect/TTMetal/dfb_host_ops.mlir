// RUN: ttmlir-opt %s | FileCheck %s
// Parser round-trip for TTMetal host-side DFB ops:
//   ttmetal.create_dataflow_buffer
//   ttmetal.bind_dfb_to_kernels

// CHECK-LABEL: func.func @test_create_and_bind_dfb
func.func @test_create_and_bind_dfb() {
  // CHECK: %[[ID:.+]] = "ttmetal.create_dataflow_buffer"()
  // CHECK-SAME: num_producers = 1, num_consumers = 1
  %dfb_id = "ttmetal.create_dataflow_buffer"() <{
    core_range = #ttmetal.core_range<0x0, 1x1>,
    config = #ttmetal.dfb_config<
      entry_size = 2048,
      num_entries = 2,
      num_producers = 1,
      num_consumers = 1,
      producer_risc_mask = 1,
      consumer_risc_mask = 256,
      producer_pattern = #ttcore.dfb_access_pattern<strided>,
      consumer_pattern = #ttcore.dfb_access_pattern<strided>,
      enable_implicit_sync = false,
      data_format = #ttcore.supportedDataTypes<bf16>
    >
  }> : () -> ui32

  // CHECK: "ttmetal.bind_dfb_to_kernels"(%[[ID]])
  // CHECK-SAME: consumer_kernel = @compute_kernel
  // CHECK-SAME: producer_kernel = @reader_kernel
  "ttmetal.bind_dfb_to_kernels"(%dfb_id) <{
    producer_kernel = @reader_kernel,
    consumer_kernel = @compute_kernel
  }> : (ui32) -> ()

  return
}

// CHECK-LABEL: func.func @test_create_dfb_1p4c_blocked
func.func @test_create_dfb_1p4c_blocked() {
  // CHECK: "ttmetal.create_dataflow_buffer"()
  // CHECK-SAME: num_producers = 1, num_consumers = 4
  // CHECK-SAME: consumer_pattern = <all>
  %dfb_id = "ttmetal.create_dataflow_buffer"() <{
    core_range = #ttmetal.core_range<0x0, 1x1>,
    config = #ttmetal.dfb_config<
      entry_size = 2048,
      num_entries = 8,
      num_producers = 1,
      num_consumers = 4,
      producer_risc_mask = 1,
      consumer_risc_mask = 3840,
      producer_pattern = #ttcore.dfb_access_pattern<strided>,
      consumer_pattern = #ttcore.dfb_access_pattern<all>,
      enable_implicit_sync = false,
      data_format = #ttcore.supportedDataTypes<bf16>
    >
  }> : () -> ui32

  return
}
