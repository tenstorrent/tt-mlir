# `ttrt`

This tool is intended to be a swiss army knife for working with flatbuffers
generated by the compiler.  Its primary role is to inspect and run flatbuffer
files.  It enables the running of flatbuffer files without a front-end runtime.

## Building

```bash
source env/activate
cmake --build build -- ttrt
ttrt --help
```

## Generate a flatbuffer file

See the [ttmlir-opt](./ttmlir-opt.md) documentation for more information on how to generate a flatbuffer file.

## APIs
```bash
ttrt --help
```

### read
```bash
ttrt read --help
ttrt read --section mlir out.ttnn
ttrt read --section cpp out.ttnn
ttrt read --section version out.ttnn
ttrt read --section system-desc out.ttnn
ttrt read --section inputs out.ttnn
ttrt read --section outputs out.ttnn
ttrt read --section all out.ttnn
ttrt read --section all out.ttnn --clean-artifacts
ttrt read --section all out.ttnn --save-artifacts
ttrt read --section all /dir/of/flatbuffers
```

### run
Note: It's required to be on a system with silicon and to have a runtime enabled
build `-DTTMLIR_ENABLE_RUNTIME=ON`.

```bash
ttrt run --help
ttrt run out.ttnn
ttrt run out.ttnn --clean-artifacts
ttrt run out.ttnn --save-artifacts
ttrt run out.ttnn --loops 10
ttrt run --program-index all out.ttnn
ttrt run --program-index 0 out.ttnn
ttrt run /dir/of/flatbuffers
ttrt run /dir/of/flatbuffers --loops 10
```

### query
Note: It's required to be on a system with silicon and to have a runtime enabled
build `-DTTMLIR_ENABLE_RUNTIME=ON`.

```bash
ttrt query --help
ttrt query --system-desc
ttrt query --system-desc-as-json
ttrt query --system-desc-as-dict
ttrt query --save-artifacts
ttrt query --clean-artifacts
```

### perf
Note: It's required to be on a system with silicon and to have a runtime enabled
build `-DTTMLIR_ENABLE_RUNTIME=ON`. Also need perf enabled build `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON` with `export ENABLE_TRACY=1`.

```bash
ttrt perf --help
ttrt perf out.ttnn
ttrt perf out.ttnn --clean-artifacts
ttrt perf out.ttnn --save-artifacts
ttrt perf out.ttnn --loops 10
ttrt perf --program-index all out.ttnn
ttrt perf --program-index 0 out.ttnn
ttrt perf --device out.ttnn
ttrt perf --generate-params --perf-csv trace.csv
ttrt perf /dir/of/flatbuffers
ttrt perf /dir/of/flatbuffers --loops 10
```

## ttrt is written as a python library, so it can be used in custom python scripts

```python
import ttrt.binary

fbb = ttrt.binary.load_from_path("out.ttnn")
d = ttrt.binary.as_dict(fbb)
```

## bonus
- artifacts are saved in ttrt-artifacts directory if the option `--save-artifacts` is provided
- you can specify `SYSTEM_DESC_PATH` with the path to your ttsys file, and lit will automatically generate all the flatbuffer binaries for that system
