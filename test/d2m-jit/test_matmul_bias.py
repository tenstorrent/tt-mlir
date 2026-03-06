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
def matmul_bias_128_per_core(a: ttnn.Tensor, b: ttnn.Tensor, bias: ttnn.Tensor, out: ttnn.Tensor):
    core_y = d2m.core_idx(0)
    core_x = d2m.core_idx(1)
    
    local_a = alloc()
    local_b = alloc()
    local_c = alloc()
    
    for k in range(0, 128, 32):
        d2m.remote_load(local_a, a[core_y:core_y+128][k:k+128])
        d2m.remote_load(local_b, b[k:k+128][core_x:core_x+128])
        d2m.matmul(local_a, local_b, local_c)

    local_bias = alloc()
    local_out = alloc()
    d2m.remote_load(local_bias, bias[core_y:core_y+128][core_x:core_x+128])
    d2m.add(local_bias, local_c, local_out)
    d2m.remote_store(out[core_y:core_y+128][core_x:core_x+128], local_out)

def test_matmul_bias_128_per_core():
    a = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    b = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    bias = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    out = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    
    module = matmul_bias_128_per_core(a, b, bias, out)
    assert module is not None
    module_str = str(module)
    with open("build/test_matmul_bias.mlir", "w") as f:
        f.write(module_str)
    # Stage 5+ lowers affine.for to scf.for; final form is ttnn.generic
    assert "scf.for" in module_str or "affine.for" in module_str or "ttnn.generic" in module_str
    assert "memref.alloc" in module_str or "ttnn.generic" in module_str

if __name__ == "__main__":
    test_matmul_bias_128_per_core()
    print("Tests passed!")
