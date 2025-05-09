[![Tests][tests badge]][tests]
[![Codecov][codecov badge]][codecov]

# tt-mlir

tt-mlir is a compiler project aimed at defining MLIR dialects to abstract compute on Tenstorrent AI accelerators.
It is built on top of the [MLIR](https://mlir.llvm.org/) compiler infrastructure and targets [TTNN](https://github.com/tenstorrent/tt-metal).

For more information on the project, see https://tenstorrent.github.io/tt-mlir/.

## Project Goals

- *Generality*: Support a wide range of AI models and workloads including training
- *Scalable*: First class primitives to describe scaling to multichip systems
- *Performant*: Enable great out of the box performance
- *Tooling*: Enable human in the loop guided compiler optimization
- *Open source*: All project development is done in the open

## Links

- [Getting Started](https://docs.tenstorrent.com/tt-mlir/getting-started.html)
- [Tools](https://tenstorrent.github.io/tt-mlir/tools.html)
- [Additional Reading](https://tenstorrent.github.io/tt-mlir/additional-reading.html)

[codecov]: https://codecov.io/gh/tenstorrent/tt-mlir
[tests]: https://github.com/tenstorrent/tt-mlir/actions/workflows/on-push.yml?query=branch%3Amain
[codecov badge]: https://codecov.io/gh/tenstorrent/tt-mlir/graph/badge.svg
[tests badge]: https://github.com/tenstorrent/tt-mlir/actions/workflows/on-push.yml/badge.svg?query=branch%3Amain
