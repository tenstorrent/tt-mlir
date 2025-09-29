# Colman's Notes

## Explicit data movement
Given the input tensor shapes and the grid shape, the user can specify arbitrary indexing and parallelism, not limited to eltwise-style or matmul-style templates. 

For example, one kernel may want to parallelize over the rows of a tensor for an eltwise op.

```python
async def dm0():
    core_num = core_index() # core number in 2d grid
    start_row_tile = core_num * rows_per_core
    end_row_tile = min(start_row_tile + rows_per_core, row_tiles)
    
    for rt in range(start_row_tile, end_row_tile):
        for ct_block in range(col_tiles // granularity):
            row_slice = slice(rt*TILE_SIZE, (rt+1)*TILE_SIZE)
            col_slice = slice(ct_block*granularity*TILE_SIZE, (ct_block+1)*granularity*TILE_SIZE)
            a_in_cb.reserve()
            tx = dma(a_accessor[row_slice, col_slice], a_in_cb)
            tx.wait()
            a_in_cb.push()
            ...
```
[colman_fused_eltwise_bcast.py](./colman_fused_eltwise_bcast.py)

while another kernel may know that input C is broadcasted across rows of A and B, therefore the kernel should be parallelized over columns of A to get maximal reuse on C.
```python
async def dm0():
    core_num = core_index() # core number in 2d grid
    start_col_tile = core_num * cols_per_core
    end_col_tile = min(start_col_tile + cols_per_core, col_tiles)
    
    c_row_slice = slice(0, 1)
    for ct in range(start_col_tile, end_col_tile):
        # Reuse C across rows of A, B
        c_col_slice = slice(ct, ct+1)
        c_in_cb.reserve()
        tx = dma(c_accessor[c_row_slice, c_col_slice], c_in_cb)
        tx.wait()
        c_in_cb.push()
        for rt_block in range(row_tiles // granularity):
            """
            Since the stream indexes by tile, slicing is cleaner
            """
            row_slice = slice(rt_block*granularity, (rt_block+1)*granularity)
            col_slice = slice(ct, ct+1)
            a_in_cb.reserve()
            tx = dma(a_accessor[row_slice, col_slice], a_in_cb)
            tx.wait()
            a_in_cb.push()
            ...
```
[colman_fused_eltwise_bcast.py](./colman_fused_eltwise_bcast.py)

## Streams and CBs
I think it simplifies the programming model if the user must specify the shape of CBs which interface between datamovement and compute. Within the compute kernel, any intermediate CBs that are needed can be automatically created.

```python
a_in_cb = CircularBuffer(a_accessor, shape=(granularity,1), buffer_factor=2)
b_in_cb = CircularBuffer(b_accessor, shape=(granularity,1), buffer_factor=2)
c_in_cb = CircularBuffer(c_accessor, shape=(1,1), buffer_factor=2)
out_cb = CircularBuffer(out_accessor, shape=(granularity,1), buffer_factor=2)
```
[colman_fused_eltwise_bcast.py](./colman_fused_eltwise_bcast.py)

In this kernel, I want to do `a * b + tanh(c)`. C is a row-vector which should be broadcasted across all rows of A and B. I believe that without specifying the shape of CBs, it's unclear what the expected broadcasting is.
There's also some trouble with how to specify that C is a tile-padded and tilized row-vector which is read in tiles but implies row-broadcasted compute APIs. I'm not sure what is the best way to specify this.


When reading a tensor into a CB or writing a tensor from a CB, the user defines a slice on the tensor.
```python
row_slice = slice(rt*TILE_SIZE, (rt+1)*TILE_SIZE)
col_slice = slice(ct_block*granularity*TILE_SIZE, (ct_block+1)*granularity*TILE_SIZE)
a_in_cb.reserve()
tx = dma(a_accessor[row_slice, col_slice], a_in_cb)
tx.wait()
a_in_cb.push()
```

I believe this can be dangerous since non-tile reads of a tilized tensor are complicated and not performant.
A way to avoid this issue could be to allow tile indexing like this
```python
a_accessor = Stream(a_in, index_type=IndexType.TILE)
...

row_slice = slice(rt, rt+1)
col_slice = slice(ct_block*granularity, (ct_block+1)*granularity)
a_in_cb.reserve()
tx = dma(a_accessor[row_slice, col_slice], a_in_cb)
tx.wait()
```

This lets the user ensure that all sliced reads are performant tile reads.

However some kernels will need to do unaligned reads into a tilized tensor. Maybe the default index type is TILE and the user must specify an unaligned index type if they know they need funky indexing.

## Do we need grids of cores?
There are kernels which don't take advantage of the 2D torus topology of our grid of cores. These kernels that operate on interleaved inputs just need a sea of cores to parallelize work over. There should be a simple interface in the `pykernel_gen` decorator to give the user `num_cores` without any 2D grid shape. When the user wants a collection of cores in no specific topology, the compiler can enforce that no mcasts are used.

Other kernels will need the 2D grid of cores in order to interact with sharded tensors or use mcasts (or other topology-aware operations, like ring store-and-forward across cores).