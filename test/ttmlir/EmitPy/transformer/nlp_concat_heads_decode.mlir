// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Verifies that NLPConcatHeadsDecodeOp emits the `sub_core_grids` argument that
// tt-metal's nlp_concat_heads_decode requires when the input is sharded on a
// core grid with more than one range or a non-origin start (issue #8945). The
// `sub_core_grids` value is imbued from the input layout by the
// NLPConcatHeadsDecodeSubCoreGridsRewritePattern in the ttnn-workaround pass;
// the EmitPy conversion then emits it.

// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --convert-ttnn-to-emitpy -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python %t.mlir | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

// Height-sharded L1 input whose core grid has two ranges [(0,0)-(7,0),
// (0,1)-(3,1)] -> triggers the subcoregrids path in tt-metal.
// Shape [1, 1, 32, 64]: num_heads=12 (padded to 32), head_dim=64 -> output
// hidden = 12 * 64 = 768.
#multi_range_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <12x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,0)>, #ttnn.core_range<(0,1), (3,1)>]>>

// Height-sharded L1 input whose core grid is a single range starting at the
// origin (0,0)-(7,0) -> does NOT trigger the subcoregrids path.
// Shape [1, 1, 32, 64]: num_heads=8, head_dim=64 -> output hidden = 8 * 64 = 512.
#single_range_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <8x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,0)>]>>

// Interleaved DRAM outputs (metal always outputs DRAM-interleaved for this op).
#dram_interleaved_out_768 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 384 + d2, d3), <1x1>, memref<1x24x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#dram_interleaved_out_512 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 256 + d2, d3), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // Multi-range sharded input: sub_core_grids must be emitted as a real
  // ttnn.CoreRangeSet (not None).
  func.func @multi_range_sub_core_grids(%arg0: tensor<1x1x32x64xbf16, #multi_range_in>) -> tensor<1x1x1x768xbf16, #dram_interleaved_out_768> {
    // CHECK: ttnn.experimental.nlp_concat_heads_decode(inputs, num_heads=12,
    // CHECK-SAME: sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)), ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(3, 1))])
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) <{num_heads = 12 : ui32}> : (tensor<1x1x32x64xbf16, #multi_range_in>) -> tensor<1x1x1x768xbf16, #dram_interleaved_out_768>
    return %0 : tensor<1x1x1x768xbf16, #dram_interleaved_out_768>
  }

  // Single-origin-range sharded input: metal won't take the subcoregrids path,
  // so sub_core_grids must be emitted as None.
  func.func @none_sub_core_grids(%arg0: tensor<1x1x32x64xbf16, #single_range_in>) -> tensor<1x1x1x512xbf16, #dram_interleaved_out_512> {
    // CHECK: ttnn.experimental.nlp_concat_heads_decode(inputs, num_heads=8,
    // CHECK-SAME: sub_core_grids=None
    %0 = "ttnn.nlp_concat_heads_decode"(%arg0) <{num_heads = 8 : ui32}> : (tensor<1x1x32x64xbf16, #single_range_in>) -> tensor<1x1x1x512xbf16, #dram_interleaved_out_512>
    return %0 : tensor<1x1x1x512xbf16, #dram_interleaved_out_512>
  }
}
