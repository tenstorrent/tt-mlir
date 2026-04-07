# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


def test_ttir_dtype_maps_contains_expected_keys():
    from chisel.utils import ttir_dtype_maps
    expected = {"i32", "f32", "bf16", "f16", "i1", "i64", "f64", "si32", "ui32"}
    assert expected.issubset(set(ttir_dtype_maps.keys()))


def test_ttir_dtype_maps_values_are_torch_dtypes():
    from chisel.utils import ttir_dtype_maps
    for key, dtype in ttir_dtype_maps.items():
        assert isinstance(dtype, torch.dtype), f"{key} maps to {dtype}, not a torch.dtype"


def test_ttrt_dtype_maps_contains_expected_keys():
    from chisel.utils import ttrt_dtype_maps
    expected = {"DataType.Float32", "DataType.BFloat16", "DataType.Int32"}
    assert expected.issubset(set(ttrt_dtype_maps.keys()))


def test_ttrt_dtype_maps_values_are_torch_dtypes():
    from chisel.utils import ttrt_dtype_maps
    for key, dtype in ttrt_dtype_maps.items():
        assert isinstance(dtype, torch.dtype), f"{key} maps to {dtype}, not a torch.dtype"


def test_debug_wrap_no_debug_reraises():
    from chisel.utils import debug_wrap

    @debug_wrap(debug=False)
    def failing():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        failing()


def test_debug_wrap_passes_through():
    from chisel.utils import debug_wrap

    @debug_wrap(debug=False)
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
