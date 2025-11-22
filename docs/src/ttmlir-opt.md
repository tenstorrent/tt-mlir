# `ttmlir-opt`

`ttmlir-opt` is the Tenstorrent optimizer driver -- a command line interface (CLI) tool used to run compiler passes on `.mlir` files. It lowers Tenstorrent IR (ttir) into ttnn or tt-metal representations. This tool is central to compiler development, testing, and generating artifacts for [ttrt](ttrt.md). Common use cases include: 
* Testing compiler passes - validate tt-mlir transformations on .mlir files before committing them.
* Generating flatbuffers for ttrt - lower a ttnn module with the backend pipeline, then translate to a flatbuffer executable.
* Targeting specific hardware - use a system descriptor to ensure generated code matches the desired hardware configuration.

## Example Commands

```bash
# Lower ttir to ttnn
./build/bin/ttmlir-opt --ttir-to-ttnn-backend-pipeline test/ttmlir/Dialect/TTNN/simple_multiply.mlir

# Lower ttir to tt-metal for execution on hardware
./build/bin/ttmlir-opt --ttir-to-ttmetal-pipeline test/ttmlir/Dialect/TTNN/simple_multiply.mlir
```
