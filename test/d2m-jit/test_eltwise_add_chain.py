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

# Decorate with jit to compile (dry run)
@jit(grid=(2,2), compile_only=True, debug=True)
def eltwise_add_chain(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, out: ttnn.Tensor):
    core_y = d2m.core_idx(0)
    core_x = d2m.core_idx(1)
    
    local_a = alloc()
    local_b = alloc()
    local_c = alloc()
    scratch = alloc()
    local_out = alloc()
    
    d2m.remote_load(local_a, a[core_y:core_y+128][core_x:core_x+128])
    d2m.remote_load(local_b, b[core_y:core_y+128][core_x:core_x+128])
    d2m.remote_load(local_c, c[core_y:core_y+128][core_x:core_x+128])
    
    d2m.add(local_a, local_b, scratch) # scratch = local_a + local_b
    d2m.add(local_c, scratch, local_out) # local_out = local_c + scratch
    
    d2m.remote_store(out[core_y:core_y+128][core_x:core_x+128], local_out)

def test_eltwise_add_chain():
    # Dummy inputs for tracing shapes
    a = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    b = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    c = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    out = ttnn.zeros([1, 1, 256, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    
    module = eltwise_add_chain(a, b, c, out)
    assert module is not None
    module_str = str(module)
    with open("build/test_eltwise_add_chain.mlir", "w") as f:
        f.write(module_str)
    assert "d2m.block_index" in module_str
    assert "d2m.remote_load" in module_str
    
if __name__ == "__main__":
    test_eltwise_add_chain()
    print("Tests passed!")
