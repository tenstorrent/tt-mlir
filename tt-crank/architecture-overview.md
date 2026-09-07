# tt-crank

Compiler & Runtime frontend for Tenstorrent hardware.

## High-level design goals

- Allow direct integration with ML frameworks, e.g. PyTorch
- Allow users to write their own custom engines for running different workloads, by exposing proper APIs for compile-time and runtime
- Support executing workloads on various hardware configurations, e.g. single chip, multi-chip, multi-host, etc.
- No difference between inference and training workloads, i.e. the same engine should be able to run both types of workloads without any changes

## High-level architecture

framework integration layer:
 - provides APIs for ML frameworks to integrate with tt-crank
 - handles conversion from framework-specific IRs (e.g. PyTorch FX graph) to a common IR (e.g. TTIR) that tt-crank can understand

tt-crank engine:
 - provides APIs for compiling user workloads - take in an IR (TTIR, StableHLO, etc.) and produce an executable image
 - provides APIs for runtime execution of compiled workloads - create and manage tensors, execute compiled images, etc.
 - caching layer for compiled images - avoid recompilation of the same workloads
 - allow users to load pre-compiled flatbuffer binaries and execute them directly without going through the compilation process
 - provide support for tensor sharding (both in compile & runtime) - this will likely mean we need to extend TTIR with this (something similar to what `shardy` does for StableHLO)

tt-mlir:
  - use `tt-mlir` as backend for compilation & runtime

ttnn & tt-metal:
  - backends for tt-mlir

## Design plan

1. Create CMake project for `tt-crank` and set up basic project structure, consume `tt-mlir` as a git submodule and external project in cmake.
  - Use `https://github.com/cpp-best-practices/cmake_template` as an inspiration for setting up c++ build system to have all warnings enabled and tooling around c++.
  - The build system should be designed in such a way that the default build commands are simple
  - Also, have a flag for building with additional checks such as `clang-tidy`
  - Run unit tests in a single command
2. For using the runtime & compile API from `tt-mlir` consult `tt-forge-onnx` and `tt-xla` repos. The abstractions and unneeded complexities should not translate to our implementation. These are useful for understanding how to use the APIs, but we should design our own abstractions that are more suitable for our use case.
3. Initially, we should define and implement a simple API for compiling and executing workloads. The input for compile should be TTIR and the flatbuffer binary should be the executable image.
4. For a POC we should start integrating with PyTorch, by implementing our own torch backend.
  - Start with supporting eager mode only
    - Implement some simple subset of operations from FX -> TTIR
    - Compile the operations to flatbuffer binary and execute them using the runtime API
  - Create simple test infra for testing single pytorch ops (TT vs. CPU)
  - Add `torch.compile` support later on, once we have the basic compilation & runtime working

## Open Questions

1. How to design the sharding support? Should we copy `shardy` or can we have something better? Ideally, we should have some way of extending TTIR with sharding information, and then have `tt-mlir` handle the sharding logic during compilation.
2. How to integrate with PyTorch distributed? What are the options there, pros and cons of each approach? Should we have our own distributed runtime or can we leverage PyTorch's distributed runtime?
