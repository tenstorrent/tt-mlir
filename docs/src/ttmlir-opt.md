# `ttmlir-opt`

The `ttmlir` optimizer driver.  This tool is used to run the `ttmlir` compiler passes on a `.mlir` source files and is central to developing and testing the compiler.

## Simple Test

```bash
./build/bin/ttmlir-opt --ttir-to-ttnn-backend-pipeline test/ttmlir/Dialect/TTNN/simple_multiply.mlir
# Or
./build/bin/ttmlir-opt --ttir-to-ttmetal-backend-pipeline test/ttmlir/Dialect/TTNN/simple_multiply.mlir
```
