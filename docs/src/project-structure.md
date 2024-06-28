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
