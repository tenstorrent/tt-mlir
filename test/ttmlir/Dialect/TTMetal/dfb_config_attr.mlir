// RUN: ttmlir-opt %s | FileCheck %s
// Test parser round-trip for the #ttmetal.dfb_config attribute.

// CHECK-LABEL: module @dfb_config_1p1c_strided
// CHECK-SAME: ttmetal.test_dfb_config = #ttmetal.dfb_config<entry_size = 2048, num_entries = 2, num_producers = 1, num_consumers = 1, producer_risc_mask = 1, consumer_risc_mask = 256, producer_pattern = <strided>, consumer_pattern = <strided>, enable_implicit_sync = false, data_format = <bf16>>
module @dfb_config_1p1c_strided attributes {
  ttmetal.test_dfb_config = #ttmetal.dfb_config<
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
} {}

// CHECK-LABEL: module @dfb_config_1p4c_blocked
// CHECK-SAME: ttmetal.test_dfb_config = #ttmetal.dfb_config<entry_size = 2048, num_entries = 8, num_producers = 1, num_consumers = 4, producer_risc_mask = 1, consumer_risc_mask = 3840, producer_pattern = <strided>, consumer_pattern = <all>, enable_implicit_sync = true, data_format = <bf16>>
module @dfb_config_1p4c_blocked attributes {
  ttmetal.test_dfb_config = #ttmetal.dfb_config<
    entry_size = 2048,
    num_entries = 8,
    num_producers = 1,
    num_consumers = 4,
    producer_risc_mask = 1,
    consumer_risc_mask = 3840,
    producer_pattern = #ttcore.dfb_access_pattern<strided>,
    consumer_pattern = #ttcore.dfb_access_pattern<all>,
    enable_implicit_sync = true,
    data_format = #ttcore.supportedDataTypes<bf16>
  >
} {}
