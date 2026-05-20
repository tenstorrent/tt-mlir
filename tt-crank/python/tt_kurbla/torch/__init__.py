from . import _native  # noqa: F401  — loading the .so runs c10::register_privateuse1_backend("tt")
from ._device import register
from tt_kurbla._runtime_env import setup_tt_metal_home

setup_tt_metal_home()

register()
