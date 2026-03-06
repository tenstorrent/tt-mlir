from d2m_jit import jit

# Stub out ttnn classes just for testing the AST parsing
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

@jit(grid=(2,2), compile_only=True, debug=True)
def eltwise_add_multi_block(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    core_y = d2m.core_idx(0)
    core_x = d2m.core_idx(1)

    core_offset_y = core_y * 128
    core_offset_x = core_x * 128

    local_a = alloc()
    local_b = alloc()
    local_out = alloc()

    for y in range(0, 128, 64):
        for x in range(0, 128, 64):
            d2m.remote_load(local_a, a[core_offset_y+y:core_offset_y+y+64][core_offset_x+x:core_offset_x+x+64])
            d2m.remote_load(local_b, b[core_offset_y+y:core_offset_y+y+64][core_offset_x+x:core_offset_x+x+64])

            d2m.add(local_a, local_b, local_out)

            d2m.remote_store(out[core_offset_y+y:core_offset_y+y+64][core_offset_x+x:core_offset_x+x+64], local_out)

def test_eltwise_add_multi_block():
    a = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    b = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    out = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    
    module = eltwise_add_multi_block(a, b, out)
    assert module is not None
    module_str = str(module)
    with open("build/test_eltwise_add_multi_block.mlir", "w") as f:
        f.write(module_str)
    # d2m.generic present through all stages up to (and including) partial stage 10 conversion
    assert "d2m.generic" in module_str or "ttnn.generic" in module_str
    # Stage 9+: kernel functions extracted as symbol refs in d2m.generic threads attribute
    assert "@compute_kernel" in module_str or "ttnn.generic" in module_str
    # Stage 10+: TTKernel ops appear in kernel functions (or stage 11 produces ttnn.generic)
    assert "ttkernel" in module_str or "ttnn.generic" in module_str
    
if __name__ == "__main__":
    test_eltwise_add_multi_block()
    print("Tests passed!")
