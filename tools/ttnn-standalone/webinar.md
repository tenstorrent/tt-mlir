# Webinar: EmitC

#### Compile tt-mlir with various flags enabled:
```bash
cmake -G Ninja \
  -B build \
  -DCMAKE_CXX_COMPILER=clang++-17 \
  -DCMAKE_C_COMPILER=clang-17 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=install \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTTMLIR_ENABLE_RUNTIME=ON \
  -DTTMLIR_ENABLE_RUNTIME_TESTS=ON \
  -DTT_RUNTIME_ENABLE_PERF_TRACE=ON \
  -DTTMLIR_ENABLE_STABLEHLO=ON \
  -DTTMLIR_ENABLE_OPMODEL=ON \
  -S .

cmake --build build
```

#### Compile model and translate to C++:

```bash
# TTNN to EmitC
ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=ttrt-artifacts/system_desc.ttsys enable-optimizer=true memory-layout-analysis-enabled=true" test/ttmlir/EmitC/TTNN/models/mnist_sharded.mlir -o mnist_emitc.mlir

# Translate IR to C++
ttmlir-translate --mlir-to-cpp mnist_emitc.mlir > tools/ttnn-standalone/ttnn-standalone.cpp
```

#### Run tracy:
```bash
TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -v build/ttnn-standalone
```
