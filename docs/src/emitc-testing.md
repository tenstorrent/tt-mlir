# EmitC testing

To locally run EmitC tests:

```bash
llvm-lit -sv test/ttmlir/EmitC/TTNN  # Generate flatbuffers and .cpp files
tools/ttnn-standalone/ci_compile_dylib.py  # Compile .cpp files to shared objects
ttrt run --emitc build/test/ttmlir/EmitC/TTNN  # Run flatbuffers + shared objects and compare results
```
