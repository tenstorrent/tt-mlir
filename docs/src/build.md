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

## Lint

- `clang-tidy`: Run clang-tidy on the project
```bash
source env/activate
cmake --build build -- clang-tidy
```

## Docs

Doc dependencies: `mdbook`

```bash
source env/activate
cmake --build build -- docs
mdbook serve build/docs/book
```
