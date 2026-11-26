# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Verify TTMetal enqueue-program location serialization preserves semantic tags."""

from __future__ import annotations

import json
import re

import pytest

import _ttmlir_runtime as tt_runtime
from ttmlir.ir import Context, Location, Module
from ttmlir.passmanager import PassManager
from ttmlir.passes import ttmetal_to_flatbuffer_bin

pytestmark = pytest.mark.frontend("ttir")

# The finish location ensures the enqueue-program location is selected.
_PROFILE_REGION_MLIR = """\
module {
  func.func @test_profile_region_loc(%arg0: i32) {
    "ttmetal.enqueue_program"(%arg0) <{
      cb_ports = array<i64>,
      kernelConfigs = [
        #ttmetal.noc_config<@scalar_kernel,
          #ttmetal.core_range<0x0, 1x1>,
          #ttmetal.kernel_args<ct_args = [<scalar[0]>]>,
          dm_core = 1, noc0>
      ],
      operandSegmentSizes = array<i32: 1, 0>
    }> : (i32) -> () loc(fused<{tt.profile.region = "test.region"}>["tagged.mlir":1:1])
    "ttmetal.finish"() : () -> () loc("finish.mlir":2:2)
    return
  }
  func.func private @scalar_kernel() attributes {
    ttkernel.arg_spec = #ttkernel.arg_spec<
      ct_args = [<arg_type = scalar, operand_index = 0>]>,
    ttkernel.thread = #ttkernel.thread<noc>
  } {
    return
  }
}
"""

_PROFILE_REGION_MLIR_LEGACY = _PROFILE_REGION_MLIR.replace(
    "dm_core = 1, noc0>", "noc0>"
)


def _parse_json(raw: str) -> object:
    return json.loads(re.sub(r"\binf\b", "Infinity", re.sub(r"\bnan\b", "NaN", raw)))


def _walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _parse_module(mlir: str) -> Module:
    ctx = Context()
    loc = Location.unknown(ctx)
    with ctx, loc:
        module = Module.parse(mlir)
    pm = PassManager.parse(
        "builtin.module("
        "ttcore-register-device,"
        "ttcore-mark-functions-as-forward,"
        "ttcore-wrap-device-module"
        ")",
        ctx,
    )
    pm.run(module.operation)
    return module


@pytest.mark.parametrize("target", ["ttmetal"])
def test_ttmetal_enqueue_program_serializes_profile_region(target: str):
    """FlatBuffer Command.loc for enqueue-program retains tt.profile.region."""
    last_error = None
    module = None
    for mlir in (_PROFILE_REGION_MLIR, _PROFILE_REGION_MLIR_LEGACY):
        try:
            module = _parse_module(mlir)
            break
        except Exception as exc:  # noqa: BLE001 - retry alternate assembly syntax
            last_error = exc
    if module is None:
        raise last_error

    capsule = ttmetal_to_flatbuffer_bin(module)
    fbb = tt_runtime.binary.load_binary_from_capsule(capsule)
    data = _parse_json(fbb.as_json())

    enqueue_locs = []
    finish_locs = []
    for node in _walk(data):
        if not isinstance(node, dict):
            continue
        type_type = str(node.get("type_type", ""))
        if "EnqueueProgram" in type_type:
            enqueue_locs.append(node.get("loc"))
        if type_type.endswith("FinishCommand") or type_type == "FinishCommand":
            finish_locs.append(node.get("loc"))

    assert enqueue_locs, "expected at least one EnqueueProgramCommand"
    assert any(
        isinstance(loc, str) and 'tt.profile.region = "test.region"' in loc
        for loc in enqueue_locs
    ), enqueue_locs
    assert any(
        isinstance(loc, str) and "finish.mlir" in loc for loc in finish_locs
    ), finish_locs
