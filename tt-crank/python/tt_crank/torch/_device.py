# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

from . import _native

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
    def get_amp_supported_dtype() -> list[torch.dtype]:
        # This makes torch.amp.autocast(device_type="tt")
        return [torch.bfloat16]

    @staticmethod
    def is_initialized() -> bool:
        # Queried by torch.distributed.DeviceMesh during init_device_mesh;
        # there's no per-device init step on our side, so we're always "ready".
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def num_chips() -> int:
        """Number of physical chips behind the single logical `tt` device.

        Distinct from `device_count()`, which is fixed at 1 — one logical
        device mapped to a multi-chip mesh.
        """
        return _native.runtime_device_num_chips()

    @staticmethod
    def arch() -> str:
        """Architecture of the chips behind the `tt` device, e.g. "wormhole_b0"."""
        return _native.runtime_device_arch()

    @staticmethod
    def set_mesh_shape(rows: int, cols: int) -> None:
        """Open (or reopen) the runtime MeshDevice as (rows, cols); rows*cols
        must be in `[1, num_chips()]`. Prefer `torch.tt.init_device_mesh`
        for the common path (sets the shape and builds the DTensor mesh together).
        """
        _native.open_runtime_device_mesh(rows, cols)

    @staticmethod
    def mesh_shape() -> tuple[int, int]:
        """Current runtime MeshDevice shape as `(rows, cols)`."""
        r, c = _native.runtime_device_mesh_shape()
        return (r, c)

    @staticmethod
    def init_device_mesh(
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...] | None = None,
    ):
        """Drop-in for `torch.distributed.init_device_mesh("tt", ...)` that
        also opens the runtime MeshDevice to match.

            mesh = torch.tt.init_device_mesh((2, 2), mesh_dim_names=("dp", "tp"))
        """
        if len(mesh_shape) == 1:
            rows, cols = 1, mesh_shape[0]
        elif len(mesh_shape) == 2:
            rows, cols = mesh_shape
        else:
            raise ValueError(
                f"torch.tt.init_device_mesh: only 1D or 2D meshes supported, got {mesh_shape}"
            )
        _native.open_runtime_device_mesh(rows, cols)
        # Each mesh dim's process group derives its runtime mesh axis from its
        # own rank composition (see TTProcessGroup.cluster_axis) — no tagging.
        return torch.distributed.device_mesh.init_device_mesh(
            "tt", mesh_shape, mesh_dim_names=mesh_dim_names
        )


def register() -> None:
    global _registered
    if _registered:
        return
    torch.utils.rename_privateuse1_backend("tt")
    torch._register_device_module("tt", _DeviceModule)
    _registered = True
