# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from d2m_jit import jit

class Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

class ttnn:
    Tensor = Tensor
    bfloat16 = "bfloat16"
    TILE_LAYOUT = "TILE_LAYOUT"
    @staticmethod
    def zeros(shape, dtype, layout):
        return Tensor(shape, dtype)

# 4×4 grid, 256×256 tensor → physical shard = 64×64 elements = 2×2 tiles per core.
# Each core loads 128×128 elements = 4×4 tiles (spanning 2×2 physical shards).
# view_layout aggregates inputs: [4,4,2,2] → [2,2,4,4] (coarser virtual grid, larger shards).
# Output is NOT coarsened: each core stores only its 64×64 physical shard (2×2 tiles).
# Pairs of physical cores (0+1, 2+3) share the same virtual input shard.
@jit(grid=(4,4), compile_only=True, debug=True)
def eltwise_add_coarse_blocks(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    core_y = d2m.core_idx(0)   # 0..3
    core_x = d2m.core_idx(1)   # 0..3

    local_a = alloc()
    local_b = alloc()
    local_out = alloc()

    # Load 128 elements (4×4 tiles = 2 physical shards) from coarsened virtual input
    d2m.remote_load(local_a, a[core_y*64:core_y*64+128][core_x*64:core_x*64+128])
    d2m.remote_load(local_b, b[core_y*64:core_y*64+128][core_x*64:core_x*64+128])
    d2m.add(local_a, local_b, local_out)
    # Store only the physical shard (64 elements = 2×2 tiles) to the output
    d2m.remote_store(out[core_y*64:core_y*64+64][core_x*64:core_x*64+64], local_out)

def test_eltwise_add_coarse_blocks():
    a   = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    b   = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    out = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)

    module = eltwise_add_coarse_blocks(a, b, out)
    assert module is not None
    module_str = str(module)
    with open("build/test_eltwise_add_coarse_blocks.mlir", "w") as f:
        f.write(module_str)
    # Coarsening view_layout on inputs only: [4,4,2,2] → [2,2,4,4].
    # Output stays at physical grid [4,4,2,2] (aligned to generic op's grid).
    assert "d2m.view_layout" in module_str
    assert "d2m.remote_load" in module_str
    assert "d2m.remote_store" in module_str

if __name__ == "__main__":
    test_eltwise_add_coarse_blocks()
    print("Tests passed!")
