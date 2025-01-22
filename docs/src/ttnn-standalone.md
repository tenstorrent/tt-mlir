# `ttnn-standalone`

TTNN Standalone is a post-compile tuning/debugging tool.

Forge and third party ML models (PyTorch, Jax, ONNX, ...) can be compiled to a set of TTNN library calls in C++. This generated code can then be used outside of the compiler environment. TTNN Standalone tool offers all the scaffolding needed to run the C++ code on device (build & run scripts).

### Usage

```bash
# Compile a model to EmitC dialect => translate to C++ code => pipe to ttnn-standalone.cpp
./build/bin/ttmlir-opt --ttir-to-emitc-pipeline test/ttmlir/EmitC/TTNN/sanity_add.mlir \
| ./build/bin/ttmlir-translate --mlir-to-cpp \
> tools/ttnn-standalone/ttnn-standalone.cpp

# Change dir to `tools/ttnn-standalone` and use the `run` script to compile and run the ttnn standalone:
cd tools/ttnn-standalone
./run
```

Note: if you receive this error
```bash
-bash: ./run: Permission denied
```
running `chmod +x run` will allow the execution of the script.
