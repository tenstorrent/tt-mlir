# Project Structure

- `env`: Contains the environment setup for building project dependencies, such as LLVM and Flatbuffers
- `include/ttmlir`: Public headers for the TTMLIR library
  - `Dialect`: MLIR dialect interfaces and definitions, dialects typically follow a common directory tree structure:
    - `IR`: MLIR operation/type/attribute interfaces and definitions
    - `Passes.[h|td]`: MLIR pass interfaces and definitions
    - `Transforms`: Common MLIR transformations, typically invoked by passes
  - `Target`: Flatbuffer schema definitions.  This defines the binary interface between the compiler and the runtime
- `lib`: TTMLIR library implementation
  - `CAPI`: C API for interfacing with the TTMLIR library, note this is needed for implementing the python bindings.  Read more about it here: https://mlir.llvm.org/docs/Bindings/Python/#use-the-c-api
  - `Dialect`: MLIR dialect implementations
- `runtime`: Device runtime implementation
  - `include/tt/runtime`: Public headers for the runtime interface
  - `lib`: Runtime implementation
  - `tools/python`: Python bindings for the runtime, currently this is where `ttrt` is implemented
- `test`: Test suite
- `tools/ttmlir-opt`: TTMLIR optimizer driver

# Namespaces

- `mlir`: On the compiler side, we use the MLIR namespace for all MLIR types and operations and subnamespace for our dialects.
  - `mlir::tt`: Everything ttmlir related is underneath this namespace.  Since
    we need to subnamespace under `mlir`, just `mlir::tt` seemed better than
    `mlir::ttmlir` which feels redundant.
    - `mlir::tt::ttir`: The TTIR dialect namespace
    - `mlir::tt::ttnn`: The TTNN dialect namespace
    - `mlir::tt::ttmetal`: The TTMetal dialect namespace
    - `mlir::tt::ttkernel`: The TTKernel dialect namespace
- `tt::runtime`: On the runtime side, we use the `tt::runtime` namespace for all runtime types and operations.
  - `tt::runtime::ttnn`: The TTNN runtime namespace
  - `tt::runtime::ttmetal`: The TTMetal runtime namespace (not implemented)
