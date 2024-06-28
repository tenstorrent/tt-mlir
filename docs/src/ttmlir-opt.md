# `ttmlir-opt`

The `ttmlir` optimizer driver.  This tool is used to run the `ttmlir` compiler passes on a `.mlir` source files and is central to developing and testing the compiler.

## Generate a flatbuffer file

```bash
./build/bin/ttmlir-opt --ttir-to-ttnn-backend-pipeline --ttnn-serialize-to-binary="output=out.ttnn" test/ttmlir/Dialect/TTNN/simple_multiply.mlir
```

## Simple Test

```bash
./build/bin/ttmlir-opt --ttir-to-ttnn-backend-pipeline test/ttmlir/Dialect/TTNN/simple_multiply.mlir
# Or
./build/bin/ttmlir-opt --ttir-to-ttmetal-backend-pipeline test/ttmlir/Dialect/TTNN/simple_multiply.mlir
```
