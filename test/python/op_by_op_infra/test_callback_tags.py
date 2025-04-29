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


@pytest.mark.parametrize("key", ["add_op_0", None])
@pytest.mark.parametrize(
    "tags",
    [(False, True), (True, False)],
)
def test_callback_tags(tags, key):
    def test_simple_callback(in0: Operand, in1: Operand, builder: TTIRBuilder):
        result = builder.add(in0, in1)
        loc_id = key if key else str(builder.get_loc())
        builder.set_callback_kv(loc_id, tags)
        return result

    module, builder = compile_as_mlir_module(
        test_simple_callback, [(64, 128), (64, 128)]
    )
    module = ttir_to_ttnn(module)
    ttnn_to_flatbuffer(module, builder, "test_callback_tags.ttnn")

    # Verify the callback tag was properly set
    bin = ttrt.binary.load_binary_from_path("ttnn/test_callback_tags.ttnn")
    loc_id = key if key else str(builder.get_loc())
    fb_tag = bin.get_program_tag(loc_id)
    assert fb_tag is not None, "Failed to retrieve callback tag"
    assert (
        fb_tag.pre_op_tag,
        fb_tag.post_op_tag,
    ) == tags, f"Expected tags {tags}, got {(fb_tag.pre_op_tag, fb_tag.post_op_tag)}"
