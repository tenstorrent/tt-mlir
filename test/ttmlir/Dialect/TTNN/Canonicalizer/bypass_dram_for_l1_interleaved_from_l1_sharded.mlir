// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Tests for BypassDRAMForL1InterleavedConsumers canonicalization pattern.
// When a DRAM to_memory_config is produced from an L1-sharded tensor, all of
// its L1-interleaved consumers are rerouted to read the L1-sharded source
// directly. If no non-L1_int compute users remain, the DRAM op is erased.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

// L1 height-sharded layout.
#l1_sharded = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <2x2>,
    memref<2x3x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>,
    core_ranges = <[#ttnn.core_range<(0,0), (1,1)>]>>

// DRAM interleaved layout.
#dram_interleaved = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>,
    memref<4x3x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// L1 interleaved layout.
#l1_interleaved = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <2x2>,
    memref<2x3x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

module attributes {} {
  // Multi-consumer DRAM: %dram has one L1_int consumer (%l1) and one non-L1_int
  // consumer (%other, a DRAM→DRAM op). The pattern fires on %dram: %l1 is
  // rerouted to read %arg0 directly, %dram is preserved for %other.
  // %other is a DRAM→DRAM identity and is folded away by the identity fold.
  //
  // CHECK-LABEL: func.func @bypass_dram_for_l1_interleaved_multi_consumer
  // CHECK-DAG: "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>
  // CHECK-DAG: "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>
  func.func @bypass_dram_for_l1_interleaved_multi_consumer(
      %arg0: tensor<1x1x128x96xbf16, #l1_sharded>,
      %arg1: !ttnn.device) -> (tensor<1x1x128x96xbf16, #dram_interleaved>,
                               tensor<1x1x128x96xbf16, #l1_interleaved>) {
    %dram = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> :
        (tensor<1x1x128x96xbf16, #l1_sharded>) -> tensor<1x1x128x96xbf16, #dram_interleaved>
    // Second consumer of %dram (non-L1_int, keeps %dram alive).
    %other = "ttnn.to_memory_config"(%dram) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> :
        (tensor<1x1x128x96xbf16, #dram_interleaved>) -> tensor<1x1x128x96xbf16, #dram_interleaved>
    // This should be rerouted to read %arg0 directly.
    %l1 = "ttnn.to_memory_config"(%dram) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> :
        (tensor<1x1x128x96xbf16, #dram_interleaved>) -> tensor<1x1x128x96xbf16, #l1_interleaved>
    return %other, %l1 : tensor<1x1x128x96xbf16, #dram_interleaved>, tensor<1x1x128x96xbf16, #l1_interleaved>
  }

  // Two L1_int consumers of the same DRAM op — both are rerouted in one step
  // and the DRAM intermediate is erased. A consumer-by-consumer approach would
  // leave one unfixed after the first rerouting reduces DRAM to single-consumer.
  //
  // CHECK-LABEL: func.func @bypass_dram_two_l1_int_consumers
  // CHECK-NOT: "ttnn.to_memory_config"{{.*}}#dram
  // CHECK: "ttnn.to_memory_config"(%arg0)
  // CHECK-SAME: memory_config = #ttnn.memory_config<#l1, <interleaved>>
  // CHECK: "ttnn.to_memory_config"(%arg0)
  // CHECK-SAME: memory_config = #ttnn.memory_config<#l1, <interleaved>>
  func.func @bypass_dram_two_l1_int_consumers(
      %arg0: tensor<1x1x128x96xbf16, #l1_sharded>,
      %arg1: !ttnn.device) -> (tensor<1x1x128x96xbf16, #l1_interleaved>,
                               tensor<1x1x128x96xbf16, #l1_interleaved>) {
    %dram = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> :
        (tensor<1x1x128x96xbf16, #l1_sharded>) -> tensor<1x1x128x96xbf16, #dram_interleaved>
    %l1_a = "ttnn.to_memory_config"(%dram) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> :
        (tensor<1x1x128x96xbf16, #dram_interleaved>) -> tensor<1x1x128x96xbf16, #l1_interleaved>
    %l1_b = "ttnn.to_memory_config"(%dram) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> :
        (tensor<1x1x128x96xbf16, #dram_interleaved>) -> tensor<1x1x128x96xbf16, #l1_interleaved>
    return %l1_a, %l1_b : tensor<1x1x128x96xbf16, #l1_interleaved>, tensor<1x1x128x96xbf16, #l1_interleaved>
  }

  // Single-consumer DRAM chain: %dram has only one non-dealloc consumer (%l1).
  // The pattern fires on %dram, reroutes %l1 to read %arg0 directly, and
  // erases %dram (no remaining compute users).
  //
  // CHECK-LABEL: func.func @fold_single_consumer_dram_from_l1_sharded
  // CHECK-NOT: "ttnn.to_memory_config"{{.*}}#dram
  // CHECK: %[[L1:.*]] = "ttnn.to_memory_config"(%arg0)
  // CHECK-SAME: memory_config = #ttnn.memory_config<#l1, <interleaved>>
  func.func @fold_single_consumer_dram_from_l1_sharded(
      %arg0: tensor<1x1x128x96xbf16, #l1_sharded>) -> tensor<1x1x128x96xbf16, #l1_interleaved> {
    %dram = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> :
        (tensor<1x1x128x96xbf16, #l1_sharded>) -> tensor<1x1x128x96xbf16, #dram_interleaved>
    %l1 = "ttnn.to_memory_config"(%dram) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> :
        (tensor<1x1x128x96xbf16, #dram_interleaved>) -> tensor<1x1x128x96xbf16, #l1_interleaved>
    return %l1 : tensor<1x1x128x96xbf16, #l1_interleaved>
  }

  // Pattern does NOT fire when the L1-sharded source already has a deallocate:
  // extending its live range past the dealloc would be unsafe.
  //
  // CHECK-LABEL: func.func @no_fold_src_already_deallocated
  // CHECK: %[[DRAM:.*]] = "ttnn.to_memory_config"(%arg0)
  // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
  // CHECK: "ttnn.to_memory_config"(%[[DRAM]])
  // CHECK-SAME: memory_config = #ttnn.memory_config<#l1, <interleaved>>
  func.func @no_fold_src_already_deallocated(
      %arg0: tensor<1x1x128x96xbf16, #l1_sharded>,
      %arg1: tensor<1x1x128x96xbf16, #dram_interleaved>) -> tensor<1x1x128x96xbf16, #l1_interleaved> {
    %dram = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> :
        (tensor<1x1x128x96xbf16, #l1_sharded>) -> tensor<1x1x128x96xbf16, #dram_interleaved>
    // Deallocate the L1-sharded source — pattern must not fire.
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x128x96xbf16, #l1_sharded>) -> ()
    // A second consumer to satisfy the "other user" check.
    "ttnn.deallocate"(%dram) <{force = false}> : (tensor<1x1x128x96xbf16, #dram_interleaved>) -> ()
    %l1 = "ttnn.to_memory_config"(%dram) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> :
        (tensor<1x1x128x96xbf16, #dram_interleaved>) -> tensor<1x1x128x96xbf16, #l1_interleaved>
    return %l1 : tensor<1x1x128x96xbf16, #l1_interleaved>
  }
}
