// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

//===----------------------------------------------------------------------===//
// Real layouts lifted verbatim from the Llama-3.1-8B prefill TTNN dump on the
// residual-stream shape 1x1024x4096. These are the exact configs the residual
// L1<->DRAM round-trip uses, so these tests mirror production precisely.
//   - #res_l1_bs : L1, block_sharded (the residual's resident L1 form, layout68)
//   - #res_dram  : DRAM, interleaved  (the staging form,                layout69)
//   - #res_l1_il : L1, interleaved    (a *different* L1 config,         layout48)
//===----------------------------------------------------------------------===//
#res_l1_bs = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <8x11>, memref<4x12x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (10,7)>]>>
#res_dram = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <1x1>, memref<32x128x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#res_l1_il = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <10x11>, memref<1x38x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
// A *different* L1 block_sharded layout of the same shape (8x4 grid vs 8x11):
// same buffer type / memory layout / tile layout as #res_l1_bs but a genuinely
// different sharding -- i.e. NOT a round-trip.
#res_l1_bs2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <8x4>, memref<4x32x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (3,7)>]>>

module attributes {} {

  //===--------------------------------------------------------------------===//
  // Basic folds
  //===--------------------------------------------------------------------===//

  // A to_memory_config whose input already has the target config is a no-op and
  // folds to its input.
  // CHECK-LABEL: func.func @identity_to_memory_config
  func.func @identity_to_memory_config(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs> {
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %arg0
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    return %0 : tensor<1x1024x4096xbf16, #res_l1_bs>
  }

  // Two consecutive to_memory_config ops with NO ops between collapse into one
  // targeting the outer config (the middle config is irrelevant).
  // CHECK-LABEL: func.func @consecutive_collapse
  // CHECK-SAME: -> tensor<1x1024x4096xbf16, [[OUT:#[a-z0-9_]+]]>
  func.func @consecutive_collapse(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_il> {
    // CHECK: %[[R:.*]] = "ttnn.to_memory_config"(%arg0){{.*}}-> tensor<1x1024x4096xbf16, [[OUT]]>
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %[[R]]
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_il>
    return %1 : tensor<1x1024x4096xbf16, #res_l1_il>
  }

  //===--------------------------------------------------------------------===//
  // The production residual-stream round-trip (occurs 32x, once per decoder
  // layer): L1 block_sharded -> DRAM interleaved -> back to the same L1
  // block_sharded. At canonicalize time (which runs before TTNNDeallocate) the
  // two ops are adjacent, so the consecutive fold rewires to
  // to_memory_config(%arg0) -> #res_l1_bs and the identity fold erases it.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: func.func @residual_roundtrip_adjacent
  func.func @residual_roundtrip_adjacent(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs> {
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %arg0
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    return %1 : tensor<1x1024x4096xbf16, #res_l1_bs>
  }

  // Same round-trip but with an unrelated op physically between the two
  // to_memory_config ops. This exercises the DRAM-staging guard's "canRelax"
  // allow-path: producer parks in DRAM, consumer moves to L1, but the net
  // change (producer input #res_l1_bs -> consumer output #res_l1_bs) is a true
  // round-trip, so the guard permits the fold and the round-trip is erased.
  // CHECK-LABEL: func.func @residual_roundtrip_op_between
  func.func @residual_roundtrip_op_between(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>, %arg1: tensor<1x1024x4096xbf16, #res_l1_bs>) -> (tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_l1_bs>) {
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: %[[A:.*]] = "ttnn.abs"(%arg1)
    // CHECK: return %arg0, %[[A]]
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.abs"(%arg1) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    %2 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    return %2, %1 : tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_l1_bs>
  }

  //===--------------------------------------------------------------------===//
  // Guard: intentional DRAM staging is PRESERVED when the net config genuinely
  // changes. Here the producer input is L1 *interleaved* and the consumer
  // output is L1 *block_sharded* -- different memory layouts -- so with ops
  // between, canRelax is false and neither op folds.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: func.func @staging_preserved_op_between
  func.func @staging_preserved_op_between(%arg0: tensor<1x1024x4096xbf16, #res_l1_il>, %arg1: tensor<1x1024x4096xbf16, #res_l1_bs>) -> (tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_l1_bs>) {
    // Both to_memory_config ops must survive.
    // CHECK: "ttnn.to_memory_config"
    // CHECK: "ttnn.abs"
    // CHECK: "ttnn.to_memory_config"
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_il>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.abs"(%arg1) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    %2 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    return %2, %1 : tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_l1_bs>
  }

  // Guard, subtler case: producer input and consumer output share buffer type,
  // memory layout, and tile layout (both L1 block_sharded tile) but have a
  // DIFFERENT sharding grid -- so this is NOT a true round-trip and the DRAM
  // staging must be preserved. (A looser 3-field "canRelax" check would wrongly
  // treat this as redundant and fold the staging away; full-layout equality
  // correctly keeps both ops.)
  // CHECK-LABEL: func.func @staging_preserved_different_grid
  func.func @staging_preserved_different_grid(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>, %arg1: tensor<1x1024x4096xbf16, #res_l1_bs>) -> (tensor<1x1024x4096xbf16, #res_l1_bs2>, tensor<1x1024x4096xbf16, #res_l1_bs>) {
    // CHECK: "ttnn.to_memory_config"
    // CHECK: "ttnn.abs"
    // CHECK: "ttnn.to_memory_config"
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.abs"(%arg1) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    %2 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs2>
    return %2, %1 : tensor<1x1024x4096xbf16, #res_l1_bs2>, tensor<1x1024x4096xbf16, #res_l1_bs>
  }

  //===--------------------------------------------------------------------===//
  // Multi-use staging buffer: the DRAM intermediate feeds another consumer in
  // addition to the round-trip. The round-trip consumer still folds away, but
  // the producer to_memory_config stays alive for its other use.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: func.func @multi_use_dram_intermediate
  func.func @multi_use_dram_intermediate(%arg0: tensor<1x1024x4096xbf16, #res_l1_bs>) -> (tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_dram>) {
    // Exactly one to_memory_config remains (the producer, kept by ttnn.abs).
    // CHECK: %[[P:.*]] = "ttnn.to_memory_config"(%arg0)
    // CHECK: %[[A:.*]] = "ttnn.abs"(%[[P]])
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %arg0, %[[A]]
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    %1 = "ttnn.abs"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_dram>
    %2 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    return %2, %1 : tensor<1x1024x4096xbf16, #res_l1_bs>, tensor<1x1024x4096xbf16, #res_dram>
  }

  //===--------------------------------------------------------------------===//
  // Reverse direction (DRAM -> L1 -> DRAM) is never gated by the guard (the
  // guard only fires for producerOut==DRAM && consumerOut==L1). A DRAM->L1->DRAM
  // round-trip back to the same config folds away unconditionally.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: func.func @dram_l1_dram_roundtrip
  func.func @dram_l1_dram_roundtrip(%arg0: tensor<1x1024x4096xbf16, #res_dram>, %arg1: tensor<1x1024x4096xbf16, #res_l1_bs>) -> (tensor<1x1024x4096xbf16, #res_dram>, tensor<1x1024x4096xbf16, #res_l1_bs>) {
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: %[[A:.*]] = "ttnn.abs"(%arg1)
    // CHECK: return %arg0, %[[A]]
    %0 = "ttnn.to_memory_config"(%arg0) : (tensor<1x1024x4096xbf16, #res_dram>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    %1 = "ttnn.abs"(%arg1) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_l1_bs>
    %2 = "ttnn.to_memory_config"(%0) : (tensor<1x1024x4096xbf16, #res_l1_bs>) -> tensor<1x1024x4096xbf16, #res_dram>
    return %2, %1 : tensor<1x1024x4096xbf16, #res_dram>, tensor<1x1024x4096xbf16, #res_l1_bs>
  }
}
