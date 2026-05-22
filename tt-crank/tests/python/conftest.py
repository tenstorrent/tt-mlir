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

import tt_kurbla.torch  # noqa: E402, F401  - registers the backend


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


@pytest.fixture(scope="session")
def tt_device() -> torch.device:
    return torch.device("tt:0")
