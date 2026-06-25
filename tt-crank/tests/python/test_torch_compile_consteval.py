"""Tests for const-eval argument tagging in the `torch.compile(backend="tt")` path.

Const-eval relies on tt-mlir's ConstEvalHoist pass + runtime GlobalTensorCache,
already wired in the pinned pipeline. tt-kurbla's job is to tag the *forward*
graph's lifted weight/buffer args ``ttcore.argument_type = parameter`` so the
pass can fold weight-only subgraphs across calls; the compiler does its own
per-function dataflow over those args, so only the args need marking.

The backward graph is deliberately left all-``input``: in training the optimizer
re-versions weights every step, so weight-derived backward const-eval entries are
invalidated before they are read (~zero payoff) and mis-tagging there is the only
path to a stale-gradient wrong result. const-eval's payoff is inference / frozen
weights, on the forward.

These tests run entirely on CPU: ``_forward_parameter_roles`` is exercised
directly, and the forward/backward arg tagging is exercised end-to-end through
aot_module_simplified with ``_lower_and_compile`` stubbed to capture the roles.
"""

from unittest.mock import patch

import torch
import torch.nn as nn

from tt_kurbla.torch import _compile, _native

# Short alias for the const-eval argument-type enum (mlir::tt::ttcore::ArgumentType).
AT = _native.ArgumentType


# --- _forward_parameter_roles: placeholder classification (no device) -------

class _FakeInputInfo:
    def __init__(self, mutates_data: bool) -> None:
        self.mutates_data = mutates_data


class _FakeFwMetadata:
    def __init__(self, static_input_indices: list[int], mutates: list[bool]) -> None:
        self.static_input_indices = static_input_indices
        self.input_info = [_FakeInputInfo(m) for m in mutates]


def test_forward_roles_tag_static_args_as_parameters() -> None:
    """Lifted params/buffers (static_input_indices) become Parameter; the trailing
    user activation stays Input."""
    # 3 args: weight (static), bias (static), activation (dynamic).
    meta = _FakeFwMetadata(static_input_indices=[0, 1], mutates=[False, False, False])
    assert _compile._fw_args_roles(3, meta) == [AT.Parameter, AT.Parameter, AT.Input]


def test_forward_roles_keep_mutated_static_arg_as_input() -> None:
    """A static arg written in-place (e.g. a KV-cache buffer) must stay Input -
    freezing it into a const-eval graph would serve stale values next step."""
    # arg0 weight (frozen), arg1 mutated buffer (must NOT freeze), arg2 input.
    meta = _FakeFwMetadata(static_input_indices=[0, 1], mutates=[False, True, False])
    assert _compile._fw_args_roles(3, meta) == [AT.Parameter, AT.Input, AT.Input]


def test_forward_roles_without_metadata() -> None:
    """With no forward metadata, classification falls back to all-Input."""
    assert _compile._fw_args_roles(2, None) == [AT.Input, AT.Input]


# --- end-to-end forward/backward tagging via aot (no device) ----------------

def _capture_roles(model: nn.Module, sample: torch.Tensor, *, backward: bool) -> dict:
    """Compile ``model`` on CPU, stubbing the TTIR lowering to capture the
    ArgumentType roles assigned to each forward/backward graph placeholder.

    Returns ``{"forward": {name: role}, "backward": {name: role}}``; the backward
    entry is present only when ``backward`` is set (a tangent placeholder marks a
    captured graph as the backward one)."""
    captured: dict = {}

    def fake_lower(gm, example_inputs, roles, *, options=None):
        names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
        kind = "backward" if any(n.startswith("tangents") for n in names) else "forward"
        captured[kind] = dict(zip(names, roles))
        return lambda *args: gm(*args)

    with patch.object(_compile, "_lower_and_compile", fake_lower):
        cmodel = torch.compile(model, backend="tt")
        out = cmodel(sample)
        if backward:
            out.sum().backward()
    return captured


def test_forward_tags_weights_as_parameters_and_input_as_input() -> None:
    """The lifted weight + bias are Parameters; the trailing user activation is an
    Input."""
    roles = _capture_roles(nn.Linear(8, 4), torch.randn(2, 8), backward=False)["forward"]
    assert sum(r == AT.Parameter for r in roles.values()) == 2
    assert sum(r == AT.Input for r in roles.values()) == 1


def test_backward_graph_is_all_input() -> None:
    """The backward graph is never tagged for const-eval: every arg (saved
    activations, saved weights, and tangents) stays Input. Tagging a saved weight
    here buys ~nothing (the optimizer re-versions weights each step) and is the
    only path to a stale-gradient wrong result."""
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    bw = _capture_roles(model, torch.randn(2, 8), backward=True)["backward"]
    assert bw, "expected a backward graph to be captured"
    assert all(r == AT.Input for r in bw.values()), f"backward must be all-Input, got {bw}"
