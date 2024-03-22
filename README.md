# ttmlir

Build

```bash
make
```

## Transforms

- Linalg input
- Tilize
- Parallelize, needs to solve for:
  - Insert affine dims c0, c1 (cluster grid size) and g0, g1 (device grid size)
  - Tensor sharding (how we can legally divide the tensor)
  - Buffer alloc (manage tensor allocations, L1 vs DRAM)
  - Cost modeling of op
  - Cost modeling of data movement between ops, i.e. reshard
  - Do we want a new tensor type that explicitly annotates grid par?
- Bufferize ?
- Insert linalg.generic for all reshards
- Lower linalg.generic to tt.dispatch (essentially the same op, but with multiple region bodies)
- Lower kernel 5 ways:
  - brisc: read from DRAM / remote core i.e. matmul systolic inner dim
  - ncrisc: write to DRAM
  - triscs: unpack / math / pack
    - Affine.for lowering inside kernel body (i.e. all affine dims except c0, c1, g0, g1)
- Serialize to flatbuffer
