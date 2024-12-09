# `ttmlir-translate`

The `ttmlir-translate` translation utility. Unlike `ttmlir-opt` tool which is used to run passes within the MLIR world, `ttmlir-translate` allows us to ingest something (e.g. code) into MLIR world, and also produce something (e.g. executable binary, or even code again) from MLIR.
## Generate C++ code from MLIR

```bash
# First, let's run `ttmlir-opt` to convert to proper dialect
./build/bin/ttmlir-opt --ttir-to-emitc-pipeline test/ttmlir/Dialect/TTNN/simple_multiply.mlir -o c.mlir

# Now run `ttmlir-translate` to produce C++ code
./build/bin/ttmlir-translate --mlir-to-cpp c.mlir
```

Bonus: These two commands can be piped, to avoid writing a `mlir` file to disk, like so:
```bash
./build/bin/ttmlir-opt --ttir-to-emitc-pipeline test/ttmlir/Dialect/TTNN/simple_multiply.mlir | ./build/bin/ttmlir-translate -mlir-to-cpp
```

## Generate flatbuffer file from MLIR
```bash
# First run `ttmlir-opt` to convert to proper dialect
./build/bin/ttmlir-opt --ttir-to-ttnn-backend-pipeline test/ttmlir/Dialect/TTNN/simple_multiply.mlir -o ttnn.mlir

# Now run `ttmlir-translate` to produce flatbuffer file
./build/bin/ttmlir-translate --ttnn-to-flatbuffer ttnn.mlir -o out.ttnn
```
