# ttnn-standalone

`ttnn-standalone` is a post-compile tuning/debugging tool.

Forge and third party ML models (PyTorch, Jax, ONNX, ...) can be compiled to a set of TTNN library op calls in C++. This generated code can then be used outside of the compiler environment. `ttnn-standalone` tool offers all the scaffolding needed to run the C++ code on device (build & run scripts).

### Usage

```bash
# 1. Convert a model from TTIR dialect to EmitC dialect using ttmlir-opt
# 2. Translate the resulting EmitC dialect to C++ code using ttmlir-translate
# 3. Pipe the generated C++ code to a .cpp file
ttmlir-opt \
  --ttir-to-emitc-pipeline \
  test/ttmlir/EmitC/TTNN/sanity_add.mlir | \
ttmlir-translate \
  --mlir-to-cpp > \
  tools/ttnn-standalone/ttnn-standalone.cpp

# 1. Change dir to `tools/ttnn-standalone`
# 2. Use `run` script to compile and run the compiled binary
cd tools/ttnn-standalone
./run
```

Note: if you receive this error
```bash
-bash: ./run: Permission denied
```
running `chmod +x run` will set the execute permission on the script.
