"""Minimal `device` fixture so the DS matmul tests here run in place.

tt-metal's own conftest supplies a richer fixture of the same name, but
third_party/tt-metal is gitignored, so copying tests into that tree loses them.
This opens one device for the whole module -- every test in this directory is
read-only with respect to device configuration.
"""
import pytest


@pytest.fixture(scope="module")
def device():
    import ttnn

    dev = ttnn.CreateDevice(device_id=0, l1_small_size=0, trace_region_size=0)
    try:
        yield dev
    finally:
        ttnn.close_device(dev)
