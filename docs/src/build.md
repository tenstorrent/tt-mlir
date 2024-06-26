# Building

## Environment setup

You only need to build this once, it builds a python virtual environment with the necessary dependencies, flatbuffers, and llvm.

```bash
cmake -B env/build env
cmake --build env/build
```

## Build

```bash
source env/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

Note:
- Enable ttnn/metal runtime via (ubuntu only): `-DTTMLIR_ENABLE_RUNTIME=ON`
- Accelerate builds with ccache: `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`

## Test

```bash
source env/activate
cmake --build build -- check-ttmlir
```

## Tools

- `ttrt`: A runtime tool for parsing and executing flatbuffer binaries
```bash
source env/activate
cmake --build build -- ttrt

./build/bin/ttmlir-opt --convert-tosa-to-ttir --ttir-to-ttnn-backend-pipeline test/ttmlir/simple_eltwise_tosa.mlir
ttrt read --section version out.ttnn
ttrt read --section system-desc out.ttnn

# If runtime enabled (-DTTMLIR_ENABLE_RUNTIME=ON)
ttrt query --system-desc
ttrt query --save-system-desc n300.ttsys
```
- `clang-tidy`: Run clang-tidy on the project
```bash
source env/activate
cmake --build build -- clang-tidy
```

## Docs

Doc dependencies: `mdbook`, `doxygen`, `graphviz`

```bash
source env/activate
cmake --build build -- docs
mdbook serve build/docs/book
```
