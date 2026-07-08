# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ttnn_jit._src.ir_layout_summary import (
    parse_ir_summary,
    render_ir_summary,
    render_ir_summary_from_mlir,
)

# Minimal TTNN IR: a DRAM-interleaved input is resharded into an L1
# width-sharded layout, a matmul runs on it, then the result is reverted back
# to the DRAM layout for the function return (an output-revert reshard that the
# greedy decision trace would NOT record).
SAMPLE_IR = """
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x64>, memref<4x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
module {
  func.func @model(%arg0: tensor<128x128xbf16, #ttnn_layout>) -> tensor<128x128xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() : () -> !ttnn.device
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<128x128xbf16, #ttnn_layout>) -> tensor<128x128xbf16, #ttnn_layout1>
    %2 = "ttnn.empty"(%0) : (!ttnn.device) -> tensor<128x128xbf16, #ttnn_layout1>
    %3 = "ttnn.matmul"(%1, %2) : (tensor<128x128xbf16, #ttnn_layout1>, tensor<128x128xbf16, #ttnn_layout1>) -> tensor<128x128xbf16, #ttnn_layout1>
    "ttnn.deallocate"(%2) : (tensor<128x128xbf16, #ttnn_layout1>) -> ()
    %4 = "ttnn.to_memory_config"(%3) : (tensor<128x128xbf16, #ttnn_layout1>) -> tensor<128x128xbf16, #ttnn_layout>
    return %4 : tensor<128x128xbf16, #ttnn_layout>
  }
}
"""


def test_parses_op_layouts_skips_noise():
    s = parse_ir_summary(SAMPLE_IR)
    names = [op.op_name for op in s.ops]
    # get_device / empty / deallocate / to_memory_config are not listed as ops.
    assert names == ["ttnn.matmul"]
    assert s.ops[0].result_layout == "l1/width_sharded/1x64 cores=(0,0)-(7,7)"


def test_parses_reshards_with_producer_consumer_and_layouts():
    s = parse_ir_summary(SAMPLE_IR)
    assert len(s.reshards) == 2

    # Reshard 1: input relayout feeding the matmul.
    r0 = s.reshards[0]
    assert r0.kind == "to_memory_config"
    assert r0.producer == "input"
    assert r0.consumer == "matmul"
    assert r0.from_layout == "dram/interleaved/1x1"
    assert r0.to_layout == "l1/width_sharded/1x64 cores=(0,0)-(7,7)"
    assert r0.is_output_revert is False

    # Reshard 2: output revert back to the function's DRAM return layout - the
    # one the greedy decision trace cannot see.
    r1 = s.reshards[1]
    assert r1.producer == "matmul"
    assert r1.consumer == "return"
    assert r1.is_output_revert is True
    assert r1.to_layout == "dram/interleaved/1x1"


def test_render_lists_reshards_and_marks_output_revert():
    text = render_ir_summary_from_mlir(SAMPLE_IR)
    assert "-- Reshards (2) [from final IR] --" in text
    assert "input -> matmul:" in text
    assert "matmul -> return:" in text
    assert "(output revert)" in text
    assert "Op layouts (from final TTNN IR)" in text


def test_empty_ir_yields_empty_summary():
    text = render_ir_summary(parse_ir_summary("module { }"))
    assert "-- Reshards (0) [from final IR] --" in text
    assert "no reshards in final IR" in text
