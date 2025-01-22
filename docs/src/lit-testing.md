# Lit testing

`llvm-lit` tool is used for MLIR testing. With it you can:

```bash
# Query which tests are available
llvm-lit -sv ./build/test --show-tests

# Run an individual test:
llvm-lit -sv ./build/test/ttmlir/Dialect/TTIR/test_allocate.mlir

# Run a sub-suite:
llvm-lit -sv ./build/test/ttmlir/Dialect/TTIR
```

> See the full [llvm-lit documentation](https://llvm.org/docs/CommandGuide/lit.html) for more information.
