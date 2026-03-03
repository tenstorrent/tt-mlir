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
- **Python**: PEP 8 with black formatter (v23.x), Python 3.12+ only
- **Functions**: Bottom-up order, helpers before callers, static/anonymous namespace for .cpp
- **Namespaces**: Lowercase, avoid `using namespace`, no aliases in headers
- **Error Handling**: Early returns to reduce nesting, no alternative tokens (&& not and)

## Pattern Rewriter Error Handling
- **NEVER call `emitOpError()` inside a pattern rewriter** - causes pass to succeed while emitting diagnostics
- Inside patterns: Use `rewriter.notifyMatchFailure()` for pattern match failures
- In `runOnOperation()`: Use `op.emitOpError()` + `signalPassFailure()` for precondition checks
- Why: `emitOpError()` in a pattern returns pattern failure (not pass failure), greedy rewriter continues, pass succeeds with diagnostics, downstream crashes occur (e.g., pytest failing with `mlir::python::PyMlirContext::ErrorCapture::~ErrorCapture(): Assertion `errors.empty() && "unhandled captured errors"' failed.`)

## Additional Notes
- Use `pre-commit run --all-files` before commits
- Create GitHub issues for TODOs with format: `TODO (alias): description. Issue: #123`
- Follow LLVM coding standards: https://llvm.org/docs/CodingStandards.html
- Follow best practices: https://llvm.org/docs/ProgrammersManual.html
