"""
Top-level pytest conftest for tt-kurbla Python tests.
"""

import os
import sys

# TT_KURBLA_USE_SIMULATOR is read in a static initializer when _native.so is
# dlopened by `import tt_kurbla.torch` (see src/engine/sim_env.cpp), so it must
# be set before that import. pytest_addoption runs after conftest import, hence
# the sys.argv sniff here.
if "--sim" in sys.argv:
    os.environ["TT_KURBLA_USE_SIMULATOR"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

from tt_kurbla.torch.testing import DeviceType  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--sim",
        action="store_true",
        default=False,
        help="Route the runtime through ttsim by setting TT_KURBLA_USE_SIMULATOR=1.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "benchmark: marks a test as a benchmark (deselect with '-m \"not benchmark\"').",
    )
    config.addinivalue_line(
        "markers",
        "multichip: marks a test as requiring >= 2 physical chips "
        "(skipped when torch.tt.num_chips() < 2).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.tt.num_chips() >= 2:
        return
    skip = pytest.mark.skip(reason="requires >= 2 physical chips")
    for item in items:
        if item.get_closest_marker("multichip") is not None:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def fixed_seed() -> None:
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def tt_device() -> torch.device:
    return torch.device("tt:0")

@pytest.fixture(scope="session")
def device_type() -> DeviceType:
    """``DeviceType.SIM`` when the runtime is routed through ttsim,
    ``DeviceType.REAL`` on silicon. Tests that need to branch per-platform
    (skip cases that hit ttsim quirks, loosen tolerances, etc.) take this
    fixture and inspect it.
    """
    return DeviceType.SIM if os.environ.get("TT_KURBLA_USE_SIMULATOR") == "1" else DeviceType.REAL


@pytest.fixture(autouse=True)
def _reset_dynamo_between_tests() -> None:
    """Wipe dynamo's compile cache before every test.

    Parametrized compile-mode tests retrace the same ``forward`` code object
    with different input dtypes/shapes — each counts as a recompile against
    ``torch._dynamo.config.recompile_limit`` (default 8). Once the limit is
    hit dynamo silently runs subsequent calls eagerly, masking real
    compile-path bugs as passing tests. Resetting per test keeps each one
    starting from a clean recompile count.
    """
    torch._dynamo.reset()


@pytest.fixture(autouse=True)
def _reset_runtime_mesh() -> None:
    """Return the process-global runtime MeshDevice to single-device after any
    test that opened a multi-chip mesh.

    Multichip tests open a mesh via ``torch.tt.init_device_mesh``; the runtime
    mesh persists across tests, so without this it leaks into later single-chip
    tests (model/op tests open no mesh of their own and would inherit it). The
    guard skips the reopen for the common case where nothing touched the mesh.
    """
    yield
    if torch.tt.mesh_shape() != (1, 1):
        torch.tt.set_mesh_shape(1, 1)


@pytest.fixture(scope="session")
def tt_pg() -> None:
    """Init the tt-backed c10d process group with world_size = num_chips().

    DTensor's collective machinery dispatches to the "tt" backend; multichip
    tests request this fixture to set it up. Single-chip tests never request it,
    so the baseline path stays PG-free.
    """
    import torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore

    n = torch.tt.num_chips()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", str(n))
    if not dist.is_initialized():
        dist.init_process_group(backend="tt", rank=0, world_size=n, store=FakeStore())
    yield
