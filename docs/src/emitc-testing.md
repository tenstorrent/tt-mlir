# EmitC testing

To locally run EmitC tests:

```bash
# Generate flatbuffers and .cpp files
llvm-lit -sv test/ttmlir/EmitC/TTNN

# Compile .cpp files to shared objects
tools/ttnn-standalone/ci_compile_dylib.py

# Run flatbuffers + shared objects and compare results
ttrt run --emitc build/test/ttmlir/EmitC/TTNN
```
