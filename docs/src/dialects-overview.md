# Dialects Overview

Here is a brief overview of the dialects in the project, please refer to the
individual dialect documentation for more details.:

- `tt`: Common types such as, `tt.tile`, `tt.metal_layout`, `tt.grid`, etc. and enums such as, data formats, memory spaces, iterator types etc.
- `ttir`: A high level dialect that models the tensor compute graph on tenstorrent devices. Accepts `tosa` and `linalg` input.
  - `ttir.generic`: Generically describe compute work.
  - `ttir.to_layout`: Convert between different tensor memory layouts and transfer between different memory spaces.
  - `tensor.pad`: Pad a tensor with a value (ie. convs)
  - `ttir.yield`: return result memref of computation in dispatch region body, lowers to `ttkernel.yield`
  - `ttir.kernel`: lowers to some backend kernel
- `ttnn`: A TTNN dialect that models ttnn API.
- `ttkernel`: Tenstorrent kernel library operations.
  - `ttkernel.noc_async_read`
  - `ttkernel.noc_async_write`
  - `ttkernel.cb_push_back`
  - `ttkernel.[matmul|add|multiply]`: Computations on tiles in source register space, store the result in dest register space.
  - `ttkernel.sfpu_*`: Computations on tiles in dest register space using sfpu coprocessor.
- `ttmetal`: Operations that dispatch work from host to device.
  - `ttmetal.dispatch`: Dispatch a grid of compute work.
