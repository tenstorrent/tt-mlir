import torch

# c10::register_privateuse1_backend("tt") already ran when _native.so loaded
# (see src/torch/_native.cpp). That establishes the canonical backend name in
# c10. This module layers Python/framework-level integration on top:
#
#   - torch.utils.rename_privateuse1_backend("tt") wires up Tensor.tt(),
#     torch.tt as an accessor, and — load-bearing — the dispatch-key alias
#     that makes `tensor.to("tt")` / `_to_copy` work in autograd-aware paths.
#     Without it, the first cross-device op surfaces an unhelpful error from
#     deep inside c10.
#   - torch._register_device_module("tt", _DeviceModule) backs the
#     `torch.tt.is_available()` / `device_count()` / `current_device()`
#     accessors that torch.utils and downstream code consult.
#
# rename_privateuse1_backend() internally calls the same c10 primitive that
# _native.cpp uses, so the C++-then-Python order enforced by __init__.py makes
# the rename a no-op for the name itself; the value-add is the Python-side
# ergonomics above.

_registered = False


class _DeviceModule:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def current_device() -> int:
        return 0


def register() -> None:
    global _registered
    if _registered:
        return
    torch.utils.rename_privateuse1_backend("tt")
    torch._register_device_module("tt", _DeviceModule)
    _registered = True
