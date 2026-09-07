# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

import tt_kurbla.torch  # noqa: F401  — import registers the backend


def test_native_loaded():
    from tt_kurbla.torch import _native

    assert _native.loaded()


def test_device_registered():
    # After c10::register_privateuse1_backend("tt") in src/torch/_native.cpp,
    # torch maps "tt" as the PrivateUse1 backend name; device.type reflects that.
    assert torch.device("tt").type == "tt"
    assert torch.tt.is_available()
    assert torch.tt.device_count() == 1
