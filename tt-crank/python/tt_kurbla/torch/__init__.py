import atexit

import torch  # noqa: F401  — load libtorch before importing the _native extension

from . import _native  # noqa: F401  — loading the .so runs c10::register_privateuse1_backend("tt")
from ._device import register

# Close the process-wide MeshDevice before interpreter finalization. Python's
# atexit runs while threads and thread-locals are still alive, unlike a
# C-runtime atexit handler.
atexit.register(_native.close_runtime_device_mesh)
from tt_kurbla._runtime_env import setup_tt_metal_home

setup_tt_metal_home()

register()

# Register the "tt" c10d distributed backend. Must come after `register()`
# above (PrivateUse1 → "tt" rename) so the backend's device list matches.
from . import _distributed  # noqa: E402, F401

from ._sharding import register_sharding_strategies  # noqa: E402

# Register our own DTensor sharding strategies for ops torch doesn't ship one
# for (index_copy).
register_sharding_strategies()

# Multi-chip APIs live on the torch.tt namespace (see _device.py): num_chips,
# init_device_mesh — that's the single front door.

# Self-registers the "tt" dynamo backend so `torch.compile(model, backend="tt")`
# works without an extra import on the user's side. Must come after `register()`
# above — the backend constructs tt-device tensors at runtime and relies on the
# PrivateUse1 rename being in place.
from . import _compile  # noqa: E402, F401
