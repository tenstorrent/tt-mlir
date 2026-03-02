
# Adds three tensors together. Only supports 256x256 sized tensors
@ttnn.d2m.jit(grid=(2,2))
def eltwise_add_chain(Tensor a, Tensor b, Tensor c, Tensor out):
  core_y = d2m.core_idx(0)
  core_x = d2m.core_idx(1)
 
  local_a = alloc()
  local_b = alloc()
  local_c = alloc()
  scratch = alloc()
  local_out = alloc()
  d2m.remote_load(local_a,a[core_y:core_y+128][core_x:core_x+128])
  d2m.remote_load(local_b,b[core_y:core_y+128][core_x:core_x+128])
  d2m.remote_load(local_c,c[core_y:core_y+128][core_x:core_x+128])
    
  d2m.add(local_a,local_b,scratch) # scratch = local_a + local_b
  d2m.add(local_c,scratch,local_out) # local_out = local_c + scratch
  
  d2m.remote_store(out[core_y:core_y+128][core_x:core_x+128], local_out)

# Matmul plus bias operation. out =  a @ b + bias. Only supports output size of 128x128.
@ttnn.d2m.jit(grid=(2,2))
def matmul_bias_128_per_core(Tensor a, Tensor b, Tensor bias, Tensor out):
  core_y = d2m.core_idx(0)
  core_x = d2m.core_idx(1)
 
  local_a = alloc()
  local_b = alloc()
  local_c = alloc()
  for k in range(0, a.dim(1), 32):
    d2m.remote_load(local_a,a[core_y:core_y+128][k:k+128])
    d2m.remote_load(local_b,b[k:k+128][core_x:core_x+128])
    
    d2m.matmul(local_a,local_b,local_c)

  local_bias = alloc()
  local_out = alloc()
  d2m.remote_load(local_bias,bias[core_y:core_y+128][core_x:core_x+128])
  d2m.add(local_bias, local_c, local_out)
  d2m.remote_store(out[core_y:core_y+128][core_x:core_x+128], local_out)

# Fused Elementwise operations: out = max(a + b, a * b)
# Assuming 128x128 grid size distributed across a 2x2 grid
@ttnn.d2m.jit(grid=(2,2))
def fused_eltwise_max_add_mul(Tensor a, Tensor b, Tensor out):
  core_y = d2m.core_idx(0)
  core_x = d2m.core_idx(1)
  
  local_a = alloc()
  local_b = alloc()
  local_add = alloc()
  local_mul = alloc()
  local_out = alloc()
  
  d2m.remote_load(local_a, a[core_y*64:core_y*64+64][core_x*64:core_x*64+64])
  d2m.remote_load(local_b, b[core_y*64:core_y*64+64][core_x*64:core_x*64+64])
  
  # Fused operations within the same d2m.generic
  d2m.add(local_a, local_b, local_add)
  d2m.mul(local_a, local_b, local_mul)
  d2m.max(local_add, local_mul, local_out)
  
  d2m.remote_store(out[core_y*64:core_y*64+64][core_x*64:core_x*64+64], local_out)

# Elementwise addition looping over multiple blocks per core
# Suppose tensors are 256x256, mapping to a 2x2 grid. 
# Each core handles 128x128 items. 
# Inside the core, we loop over four 64x64 sub-blocks.
@ttnn.d2m.jit(grid=(2,2))
def eltwise_add_multi_block(Tensor a, Tensor b, Tensor out):
  core_y = d2m.core_idx(0)
  core_x = d2m.core_idx(1)

  # Each core operates on a 128x128 sub-tensor
  core_offset_y = core_y * 128
  core_offset_x = core_x * 128

  local_a = alloc()
  local_b = alloc()
  local_out = alloc()

  for y in range(0, 128, 64):
    for x in range(0, 128, 64):
      # Load a 64x64 block
      d2m.remote_load(local_a, a[core_offset_y+y:core_offset_y+y+64][core_offset_x+x:core_offset_x+x+64])
      d2m.remote_load(local_b, b[core_offset_y+y:core_offset_y+y+64][core_offset_x+x:core_offset_x+x+64])

      d2m.add(local_a, local_b, local_out)

      # Store the 64x64 block
      d2m.remote_store(out[core_offset_y+y:core_offset_y+y+64][core_offset_x+x:core_offset_x+x+64], local_out)



