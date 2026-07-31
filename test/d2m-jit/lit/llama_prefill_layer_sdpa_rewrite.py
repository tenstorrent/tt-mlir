# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""Check fused SDPA matching against the full captured Llama prefill layer."""

from pathlib import Path
import sys


def find_tt_mlir_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    root = find_tt_mlir_root()
    sys.path.insert(0, str(root / "test" / "d2m-jit"))

    from d2m_jit._src.rewrite import apply_patterns_text

    pattern = (
        root
        / "test"
        / "d2m-jit"
        / "kernels"
        / "patterns"
        / "llama_prefill_sdpa_to_kernel.py"
    )
    model = (
        root
        / "test"
        / "ttmlir"
        / "models"
        / "single_blocks_and_layers"
        / "llama_3_8b_prefill_layer.mlir"
    )

    rewritten = apply_patterns_text(model.read_text(), [str(pattern)])
    generic_count = rewritten.count("d2m.generic")
    if generic_count != 5:
        raise AssertionError(
            f"expected four cache and one SDPA generic, found {generic_count}"
        )
    print(rewritten)


if __name__ == "__main__":
    main()


# CHECK-LABEL: func.func @main
# CHECK-COUNT-4: d2m.generic {{.*}}grid = #ttcore.grid<8x8>
# CHECK: d2m.generic
# CHECK-SAME: grid = #ttcore.grid<8x4>
# CHECK: d2m.tile_matmul
# CHECK: d2m.tile_reduce_max
# CHECK: d2m.tile_exp
# CHECK: d2m.tile_reduce_sum
# CHECK: d2m.tile_bcast
