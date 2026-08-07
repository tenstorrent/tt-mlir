"""The artifacts dumper end to end: compile MNISTLinear three ways and read back
what landed on disk.

Covers the whole chain in one go - `capture_ttir` reaching the engine, the
forward/backward tag, the per-graph file names, and the `artifacts.json` index.
"""

import contextlib
import json
from pathlib import Path

import pytest
import torch

from tt_kurbla.torch import _artifacts
from tt_kurbla.torch._artifacts import collect_artifacts

from _models import MNISTLinear


_BATCH = 32
_FEAT = 28 * 28
_HIDDEN = 32
_CLASSES = 10

# bf16 is the one dtype both silicon and ttsim execute.
_DTYPE = torch.bfloat16


@pytest.fixture
def artifacts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect dumps into `tmp_path`.

    Patches `_artifacts_root` rather than setting TT_KURBLA_ARTIFACTS_DIR: the C++
    side reads the environment in a static initializer when the extension loads, so
    setting it from a test would have no effect.
    """
    monkeypatch.setattr(_artifacts, "_artifacts_root", lambda: tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "no_grad,run_backward,expected_graphs",
    [
        (True, False, ["graph_0_inference"]),
        (False, False, ["graph_0_forward"]),
        (False, True, ["graph_0_forward", "graph_1_backward"]),
    ],
    ids=["inference", "forward", "training"],
)
def test_artifacts_dump(
    artifacts_root: Path,
    tt_device: torch.device,
    no_grad: bool,
    run_backward: bool,
    expected_graphs: list[str],
) -> None:
    model = torch.compile(MNISTLinear(_FEAT, _HIDDEN, _CLASSES).to(tt_device, dtype=_DTYPE), backend="tt")
    x = torch.randn(_BATCH, _FEAT, device=tt_device, dtype=_DTYPE)

    with collect_artifacts("mnist_linear"):
        with torch.no_grad() if no_grad else contextlib.nullcontext():
            out = model(x)
        # aot compiles the backward lazily, on the first backward call - so this has
        # to happen inside the collection to be seen.
        if run_backward:
            out.sum().backward()

    (out_dir,) = list(artifacts_root.iterdir())
    assert out_dir.name.startswith("mnist_linear_")

    index = json.loads((out_dir / "artifacts.json").read_text())
    assert index["collection"] == "mnist_linear"
    assert [graph["graph"] for graph in index["graphs"]] == expected_graphs

    for graph in index["graphs"]:
        assert "func.func" in (out_dir / graph["ttir"]).read_text()
        assert "ttnn." in (out_dir / graph["ttnn"]).read_text()
        assert graph["compile_options"]
        # Not asserted `False`: the compile cache is process-wide, so an earlier
        # test in the same session may legitimately have compiled this graph.
        assert isinstance(graph["cache_hit"], bool)
