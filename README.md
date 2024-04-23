# ttmlir

## Build

```bash
make
source env/activate
```

## This repo is very much a work in progress

- mlir input tosa/linalg examples `test/ttmlir`.
- scratch.mlir is the scratch pad space I'm using for outlining the design of the dialect.
- Simple linalg test `./build/bin/ttmlir-opt --tt-tilize --tt-parallelize --tt-lower --tt-code-gen test/ttmlir/simple_eltwise_linalg.mlir`
- Simple tosa test still WIP, pass order is still TBD.

## Useful links / reading

- [affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
  - Affine map is a really powerful primitive that can be used to describe most data movement patterns.
  - It can also be used to describe memory layouts.
- [linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [tosa dialect](https://mlir.llvm.org/docs/Dialects/TOSA/)
- [memref dialect](https://mlir.llvm.org/docs/Dialects/MemRef/)
- [tosa spec](https://www.mlplatform.org/tosa/tosa_spec.html)
- [torch-mlir](https://github.com/llvm/torch-mlir)
  - See `python/resnet18.py` for an example of collecting torch-mlir output.
  - `source env/activate && python python/resnet18.py > resnet18_linalg.mlir`
  - Uncomment `output_type = "linalg-on-tensors"` to get linalg output.
  - Uncomment `resnet50` to get resnet50 output.
  - Tosa dialect by default embeds all constant tensor data + weights in the IR making it too large to checkin. Hopefully there's a way to disable this.

## Types

- `ttmlir::TileType`: All of the various TT tile types.
- `ttmlir::TensorType`: A tensor type with shape and element type.

## Ops

- tt.layout: A layout op. This op implements tensor layout conversions, where layout refers to how the tensor data is physically laid out in memory and sliced accross cores.
- tt.dispatch: Dispatch a kernel to a device grid. Has only a few flavors:
  - `tt.builtin`: A builtin kernel. This is a kernel that is written by hand / implemented in the runtime.
  - `linalg.generic`: A generic linalg op. The compiler generates a kernel for this op.
- tt.barrier: TBD. A barrier op. This op is used to synchronize the host and device.

## Proposed Transforms

- Tosa/Linalg input
- High level optimization passes
  - e.g. constant folding `createTosaLayerwiseConstantFoldPass`
  - Transpose / reshape cancelation
- Lower tosa/linalg to tt.dispatch
- Lower mlir.tensor to tt.tensor, see scratch.mlir for some potential tt.tensor ideas
- Op Fusion, hoisting ops into the region body of a dispatch op.
- Parallelize, needs to solve for:
  - Insert affine dims d0, d1 (cluster device indices) and c0, c1 (device core indices)
  - Tensor blocking (how we can legally divide the tensor) (described by affine maps)
  - Buffer alloc (manage tensor allocations, L1 vs DRAM)
  - Cost modeling of op
  - Cost modeling of tt.layout (reblocking)
- Automatic generation of data movement kernels
- Loading the tt.builtin kernels from source, i.e. Tosa path
- Automatic generation of compute kernels for linalg.generic ops
- Serialize to flatbuffer
  - Eventually there will be a runtime subdir that can execute the flatbuffer
  - The flatbuffer can be trivially serialized to disk and deployed
  - Flatbuffer API is all C++, so it should be easy to integrate in embedded systems
  - See .fbs files for an outline of the idea

## Opens

- How to represent pipeline parallelism?

## Tried

- Vectorization
  - This pass automatically inserts outer loops, ideally we'd want to stay affine longer
- Tiling
  - This pass also automatically inserts loops
