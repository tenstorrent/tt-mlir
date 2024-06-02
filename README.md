# ttmlir

## Build

```bash
make
source env/activate
```

## This repo is very much a work in progress

- mlir input tosa/linalg examples `test/ttmlir`.
- Simple test (each flag is a pass that runs in order):
```bash
./build/bin/ttmlir-opt --convert-tosa-to-ttir --ttir-dispatch --ttir-layout --ttir-allocate --convert-ttir-to-ttmetal --ttmetal-serialize-to-binary -mlir-print-stacktrace-on-diagnostic test/ttmlir/simple_eltwise_tosa.mlir
```

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

## Dialects

- `tt`: Common types such as, `tt.tile`, `tt.layout`, `tt.grid`, etc. and enums such as, data formats, memory spaces, iterator types etc.
- `ttir`: A high level dialect that models the tensor compute graph on tenstorrent devices. Accepts `tosa` and `linalg` input.
  - `ttir.dispatch`: Dispatch a grid of compute work.
  - `ttir.layout`: Convert between different tensor memory layouts and transfer between different memory spaces.
  - `tensor.pad`: Pad a tensor with a value (ie. convs)
  - `ttir.yield`: return result memref of computation in dispatch region body, lowers to `ttkernel.yield`
  - `ttir.kernel`: lowers to some backend kernel
- `ttnn`: A TTNN dialect that pattern matches to ttnn operations.
- `ttir` lowers to `ttnn` or `ttmetal` and `ttkernel` dialects:
  - `ttmetal`: Operations that dispatch work from host to device.
    - `ttmetal.dispatch`: Dispatch a grid of compute work.
  - `ttkernel`: Tenstorrent kernel library operations.
    - `ttkernel.ffi`: Call a function defined outside of this compilation context.  Usually a hand written piece of C++.
    - `ttkernel.noc_async_read`
    - `ttkernel.noc_async_write`
    - `ttkernel.cb_push_back`
    - `ttkernel.[matmul|add|multiply]`: Computations on tiles in source register space, store the result in dest register space.
    - `ttkernel.sfpu_*`: Computations on tiles in dest register space using sfpu coprocessor.
