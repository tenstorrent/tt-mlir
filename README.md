# ttmlir

Build

```bash
make
```

## Types

- `ttmlir::TileType`: All of the various TT tile types.
- `ttmlir::TensorType`: A tensor type with shape and element type.

## Ops

- tt.layout: A layout op. This op implements tensor layout conversions, where layout refers to how the tensor data is physically laid out in memory and sliced accross cores.
- tt.dispatch: Dispatch a kernel to a device grid. Has only a few flavors:
  - `tt.builtin`: A builtin kernel. This is a kernel that is written by hand / implemented in the runtime.
  - `tt.layout`: TBD
  - `linalg.generic`: A generic linalg op. The compiler generates a kernel for this op.
- tt.barrier: A barrier op. This op is used to synchronize the host and device.

## Transforms

- Tosa/Linalg input
- Tilize
- Lower tosa/linalg.generic to tt.dispatch (essentially the same op, but with multiple region bodies)
- Lower mlir.tensor to tt.tensor
- Parallelize, needs to solve for:
  - Insert affine dims c0, c1 (cluster grid size) and g0, g1 (device grid size)
  - Tensor sharding (how we can legally divide the tensor)
  - Buffer alloc (manage tensor allocations, L1 vs DRAM)
  - Cost modeling of op
  - Cost modeling of data movement between ops, i.e. reshard
  - Do we want a new tensor type that explicitly annotates grid par?
- Lower kernel 5 ways:
  - brisc: read from DRAM / remote core i.e. matmul systolic inner dim
  - ncrisc: write to DRAM
  - triscs: unpack / math / pack
    - Affine.for lowering inside kernel body (i.e. all affine dims except c0, c1, g0, g1)
- Serialize to flatbuffer

## Tried

- Vectorization
  - This pass automatically inserts outer loops, ideally we'd want to stay affine longer
- Tiling
  - This pass also automatically inserts loops
