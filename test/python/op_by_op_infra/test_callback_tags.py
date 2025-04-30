# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import pytest

from ttmlir.test_utils import (
    compile_as_mlir_module,
    ttnn_to_flatbuffer,
    ttir_to_ttnn,
    compile_to_flatbuffer,
)
from ttmlir.ttir_builder import Operand, TTIRBuilder
import ttrt.binary


@pytest.mark.parametrize(
    "default_tags",
    [(False, True), None],
)
@pytest.mark.parametrize("pre_op_tag", [True, False])
@pytest.mark.parametrize("post_op_tag", [True, False])
def test_callback_tags(default_tags, pre_op_tag, post_op_tag):
    def test_simple_callback(in0: Operand, in1: Operand, builder: TTIRBuilder):
        add_1 = builder.add(in0, in1)
        add_2 = builder.add(in0, add_1)
        add_3 = builder.add(add_1, add_2)
        if default_tags:
            builder.set_default_pre_op_callback_tag(default_tags[0])
            builder.set_default_post_op_callback_tag(default_tags[1])
        builder.set_operand_pre_op_callback_tag(add_1, pre_op_tag)
        builder.set_operand_post_op_callback_tag(add_2, post_op_tag)
        return add_3

    module, builder = compile_as_mlir_module(
        test_simple_callback, [(64, 128), (64, 128)]
    )
    module = ttir_to_ttnn(module)
    ttnn_to_flatbuffer(module, builder, "test_callback_tags.ttnn")

    # Verify the callback tags were properly set
    bin = ttrt.binary.load_binary_from_path("ttnn/test_callback_tags.ttnn")
    op_dict = builder.get_operand_id_map()
    builder_tags = builder.get_callback_map()
    for op, loc in op_dict.items():
        builder_tag = builder_tags[str(loc)]
        fb_tag = bin.get_tag_by_name(str(loc))
        assert builder_tag is not None, "Failed to retrieve builder callback tag"
        assert fb_tag is not None, "Failed to retrieve flatbuffer callback tag"
        assert (fb_tag.name, fb_tag.pre_op_tag, fb_tag.post_op_tag,) == (
            builder_tag.name,
            builder_tag.pre_op_tag,
            builder_tag.post_op_tag,
        ), f"Expected tags {(builder_tag.pre_op_tag, builder_tag.post_op_tag)}, got {(fb_tag.pre_op_tag, fb_tag.post_op_tag)}"
