# TT-MLIR Agent Guidelines

## Build/Lint/Test Commands
- **Environment**: `source env/activate` (activate virtual environment first)
- **Configure**: `cmake -G Ninja -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache`
- **Build**: `cmake --build build`
- **Lint**: `pre-commit run --all-files` (includes clang-format, black, copyright checks)
- **Compiler tests**: `cmake --build build --target check-ttmlir`
- **Single MLIR test**: `llvm-lit test/ttmlir/path/to/test.mlir`
- **Python tests**: `pytest test/python` (requires `ttrt query --save-artifacts` and `SYSTEM_DESC_PATH`)
- **Performance tests**: `cmake --build build --target check-perf` (requires `-DTTMLIR_ENABLE_OPMODEL=ON`)

## Code Style Guidelines
- **C++ Style**: LLVM style (see .clang-format, .clang-tidy)
- **Naming**: UpperCamelCase for types, lowerCamelCase for variables/functions
- **Includes**: Absolute paths from ttmlir root, sorted: main header → local → LLVM → system
- **Comments**: Full sentences, explain why not what, TODO with alias and issue link
- **Python**: PEP 8 with black formatter (v23.x), Python 3.10+ only
- **Functions**: Bottom-up order, helpers before callers, static/anonymous namespace for .cpp
- **Namespaces**: Lowercase, avoid `using namespace`, no aliases in headers
- **Error Handling**: Early returns to reduce nesting, no alternative tokens (&& not and)

## Pass Development Guidelines
- **Precondition validation**: Check preconditions in `runOnOperation()` before pattern application
  - Scan the IR for invalid state (e.g., unconverted linalg ops before DST allocation)
  - Emit diagnostic with `emitOpError()` on the specific problematic operation
  - Call `signalPassFailure()` immediately and return
  - Do NOT defer errors through pattern state or atomic flags (thread-safety)
- **Pattern rewrites**: Keep simple, use `notifyMatchFailure()` for non-matches
- **Error messages**: Make them actionable (e.g., "run --d2m-linalg-to-affine before this pass")

## Additional Notes
- Use `pre-commit run --all-files` before commits
- Create GitHub issues for TODOs with format: `TODO (alias): description. Issue: #123`
- Follow LLVM coding standards: https://llvm.org/docs/CodingStandards.html
