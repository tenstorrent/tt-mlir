## Table of contents

- [TTNN Standalone](#ttnn-standalone)
  - [Usage](#usage)
- [TTNN Dylib](#ttnn-dylib)

## TTNN Standalone

TTNN Standalone is a post-compile tuning tool.

Third party ML models (PyTorch, Jax, ONNX, ...) can be compiled to a set of TTNN library calls in C++. This generated code can then be manually fine-tuned outside of the compiler environment. TTNN Standalone tool offers all the scaffolding needed to run the C++ code on device (build & run scripts).

### Usage

```bash
# Compile a model to C++ code
./build/bin/ttmlir-opt --ttir-to-emitc-pipeline test/ttmlir/Silicon/TTNN/emitc/simple_add.mlir | ./build/bin/ttmlir-translate --mlir-to-cpp

# Copy paste the generated function into `ttnn-standalone.cpp`.

# Adapt the `main()` function in `ttnn-standalone.cpp` to feed tensors needed for the model

# Run the following script from within this folder (`tools/ttnn-standalone`) to compile and run the ttnn standalone:

./run
```

Note: if you receive this error
```bash
-bash: ./run: Permission denied
```
running `chmod +x run` will allow the execution of the script.

## TTNN Dylib

Similarly to the Standalone, this tool offers the ability to compile third party ML models, but to dylibs. Initial intent for compiled dylibs is to be used in testing infrastructure, but sky's the limit :)
