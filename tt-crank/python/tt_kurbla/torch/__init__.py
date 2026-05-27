from . import _native  # noqa: F401  — loading the .so runs c10::register_privateuse1_backend("tt")
from ._device import register
from tt_kurbla._runtime_env import setup_tt_metal_home

setup_tt_metal_home()

register()

# Self-registers the "tt" dynamo backend so `torch.compile(model, backend="tt")`
# works without an extra import on the user's side. Must come after `register()`
# above — the backend constructs tt-device tensors at runtime and relies on the
# PrivateUse1 rename being in place.
from . import _compile  # noqa: E402, F401
