# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Check that IRModule resolves the flatbuffer program name of every control flow
region back to the block it was serialized from.

The names are a contract with `createOp(FlatbufferObjectCache &, WhileOp|CaseOp,
...)` in lib/Target/TTNN/TTNNToFlatbuffer.cpp. These cases need no device: they
build an IRModule straight from MLIR text.
"""
import pytest

from chisel.ops import IRModule

LAYOUT_PREAMBLE = """
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#l2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#index = #ttnn.ttnn_layout<() -> (0, 0), <1x1>, memref<1x1xsi32, #system_memory>>
#pred = #ttnn.ttnn_layout<() -> (0, 0), <1x1>, memref<1x1xui32, #system_memory>>
"""

THREE_BRANCH_CASE = (
    LAYOUT_PREAMBLE
    + """
module {
  func.func @main(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<si32, #index>) -> tensor<64x128xbf16, #l2> {
    %0 = ttnn.case index(%arg1 : tensor<si32, #index>) captures(%arg0 : tensor<64x128xbf16, #l2>) branches {
    ^bb0(%cap: tensor<64x128xbf16, #l2>):
      ttnn.yield %cap : tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%cap: tensor<64x128xbf16, #l2>):
      %1 = "ttnn.add"(%cap, %cap) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      ttnn.yield %1 : tensor<64x128xbf16, #l2>
    }, {
    ^bb0(%cap: tensor<64x128xbf16, #l2>):
      %1 = "ttnn.multiply"(%cap, %cap) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
      ttnn.yield %1 : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    return %0 : tensor<64x128xbf16, #l2>
  }
}
"""
)

CASE_IN_WHILE = (
    LAYOUT_PREAMBLE
    + """
module {
  func.func @main(%arg0: tensor<64x128xbf16, #l2>, %arg1: tensor<ui32, #pred>, %arg2: tensor<si32, #index>) -> tensor<64x128xbf16, #l2> {
    %0 = ttnn.while inits(%arg0 : tensor<64x128xbf16, #l2>) captures(%arg1, %arg2 : tensor<ui32, #pred>, tensor<si32, #index>) {trip_count = 2 : i64} cond {
    ^bb0(%acc: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>, %i: tensor<si32, #index>):
      ttnn.yield %p : tensor<ui32, #pred>
    } do {
    ^bb0(%acc: tensor<64x128xbf16, #l2>, %p: tensor<ui32, #pred>, %i: tensor<si32, #index>):
      %1 = ttnn.case index(%i : tensor<si32, #index>) captures(%acc : tensor<64x128xbf16, #l2>) branches {
      ^bb0(%cap: tensor<64x128xbf16, #l2>):
        ttnn.yield %cap : tensor<64x128xbf16, #l2>
      }, {
      ^bb0(%cap: tensor<64x128xbf16, #l2>):
        %2 = "ttnn.add"(%cap, %cap) : (tensor<64x128xbf16, #l2>, tensor<64x128xbf16, #l2>) -> tensor<64x128xbf16, #l2>
        ttnn.yield %2 : tensor<64x128xbf16, #l2>
      } -> (tensor<64x128xbf16, #l2>)
      ttnn.yield %1 : tensor<64x128xbf16, #l2>
    } -> (tensor<64x128xbf16, #l2>)
    return %0 : tensor<64x128xbf16, #l2>
  }
}
"""
)


@pytest.mark.parametrize(
    "source,program_names",
    [
        pytest.param(
            THREE_BRANCH_CASE,
            [
                "main",
                "main_case_0_branch_0",
                "main_case_0_branch_1",
                "main_case_0_branch_2",
            ],
            id="three_branch_case",
        ),
        pytest.param(
            CASE_IN_WHILE,
            [
                "main",
                "main_while_0_cond",
                "main_while_0_body",
                "main_case_0_branch_0",
                "main_case_0_branch_1",
            ],
            id="case_in_while",
        ),
    ],
)
def test_region_program_names_resolve(source, program_names):
    """Every region program name maps to a block, with the right inputs.

    `branches` is a variadic region, so ODS reports a single region name for all
    of them. Deriving the per-branch suffixes from that name instead of from the
    region count would silently register only the first branch and leave the rest
    unresolvable.
    """
    ir_module = IRModule(mlir_source=source, functions=program_names)

    for name in program_names:
        # Would raise during construction if the name had not been registered.
        assert ir_module.get_function_ops(name) is not None

    # A branch program takes the captures as its inputs, so exactly one here.
    for name in program_names:
        if "_branch_" in name:
            assert len(ir_module.get_function_inputs(name)) == 1
            assert len(ir_module.get_function_outputs(name)) == 1


def test_unregistered_region_program_is_an_error():
    """A name no region produced is reported rather than silently ignored."""
    with pytest.raises(ValueError, match="main_case_0_branch_3"):
        IRModule(
            mlir_source=THREE_BRANCH_CASE,
            functions=["main", "main_case_0_branch_3"],
        )
