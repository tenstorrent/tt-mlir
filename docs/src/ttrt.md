# ttrt

ttrt (Tenstorrent Runtime) is a tool included in the tt-mlir toolchain that is designed to inspect and execute flatbuffer files produced by the Tenstorrent compiler stack--without requiring a front-end runtime. In the Tenstorrent workflow, flatbuffers are the final compiled representation of a model. They are normally consumed by a full runtime, but if you are a compiler developer, hardware engineer, or member of a bring-up team you may need to work with the flatbuffers directly. ttrt acts as a "Swiss-army knife" utility for flatbuffers, allowing you to: 
* Run them
* Debug them
* Generate artifacts (descriptors) 

The general workflow is as follows:

1. **Compile models with tt-mlir** – transforms high-level models into MLIR, applies optimization passes, and outputs the final flatbuffer file.

2. **Use ttrt** – runs the flatbuffers, debugs them, and generates artifacts such as system descriptors, logs, and test files as needed.

3. **tt-metal executes ops** – the backend library handles memory management, scheduling, and interfacing with Tenstorrent hardware when execution is required.

4. **Flatbuffers** – act as the portable, versioned model representation bridging the compiler, ttrt, tt-metal, and the hardware.

## Options for using ttrt

There are two ways you can use ttrt: 
* ttrt CLI - Build tt-mlir, then access ttrt through the command line. This document covers everything you need to work with the ttrt command line interface.
* [ttrt wheel](ttrt_apis.md) - Install what's needed to access ttrt APIs and use them in scripts. Set up is less complex.

## Prerequisites 
This section outlines all the prerequisites you need to use the ttrt CLI successfully. 

* **Hardware requirements** - 
    * You must run ttrt on a silicon-enabled system for most modes (`perf`, `emitpy`, `emitc`, `check`).
    * Host-only mode (`--host-only`) is available if device cores are unavailable
* **Build requirements** - 
    * Runtime must be enabled: `-DTTMLIR_ENABLE_RUNTIME=ON` 
    * Additionally, for perf tracing, you must enable: `DTT_RUNTIME_ENABLE_PERF_TRACE=ON`
    * clang-17 / clang++-17 is the recommended compiler (refer to tt-mlir build requirements for more details)
* **Software dependencies** - 
    * Python >= 3.10 (for `emitpy`)
    * GNU debugger (`gdb`) for debugging (`--gdb`)
    * Tracy client/server setup for advanced profiling (`--port` optional)
* **Version alignment** - 
    * ttrt binary, runtime, and flatbuffers must match 
    * System descriptor used at compile time must match the runtime system (`--system-desc`)
* **Environment variables (optional)**
    * `export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys`
    * `export TT_METAL_LOGGER_LEVEL=DEBUG  # verbose logging for tt-metal`
* **Artifact and logging setup** 
    * Recommended artifact directory: `ttrt --artifact-dir /path/to/artifacts ...`
    * Cleaning previous artifacts: `--clean-artifacts`

## Building
This section shows you how to build tt-mlir, which is required for running the ttrt CLI.

1. Use the build instructions for [tt-mlir](getting-started.md). When you reach the section [Building the tt-mlir Project](https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/getting-started.md#building-the-tt-mlir-project) you need to make a change to one of the steps. You cannot build TTRT unless you flag the build for one of: 
* Building runtime mode (ttrt fully executes the model on Tenstorrent hardware via tt-metal) - `-DTTMLIR_ENABLE_RUNTIME=ON`
* Building perf mode (ttrt measures performance, traces ops, and simulates execution, using tt-metal partially or in a non-executing mode) - `-DTTMLIR_ENABLE_RUNTIME=ON / -DTT_RUNTIME_ENABLE_PERF_TRACE=ON`

> **NOTE:** If you build tt-mlir, you need to be in the tt-mlir directory, and you can use the virtual environment created (`env`) for this tutorial as well.

The change goes in the cmake command, here is an example: 
```bash
cmake -G Ninja -B build -DTTMLIR_ENABLE_RUNTIME=ON
```

The complete build instructions for tt-mlir (add one of the modifications described above) can be found here - [Build ttmlir](./getting-started.md). 

2. Build `ttrt`:
```bash
source env/activate
cmake --build build
ttrt --help
```

## LOGGER levels
ttrt supports logging at different logger levels. You need to set env var `TTRT_LOGGER_LEVEL` in command line or a [python script](./ttrt.md#logging). By default, it is set to `INFO`.

```bash
TTRT_LOGGER_LEVEL=INFO
TTRT_LOGGER_LEVEL=CRITICAL
TTRT_LOGGER_LEVEL=ERROR
TTRT_LOGGER_LEVEL=WARNING
TTRT_LOGGER_LEVEL=DEBUG
```

### tt-metal logging
`ttrt` uses [tt-metal](https://github.com/tenstorrent/tt-metal) for op execution and device interfacing. For more detailed logs, which can help in troubleshooting build or runtime issues, set env var `TT_METAL_LOGGER_LEVEL`. By default, it is set to `FATAL`.

```bash
export TT_METAL_LOGGER_LEVEL=DEBUG
```

## Generating flatbuffers
Flatbuffers are the final, portable representation of a compiled model in the Tenstorrent workflow. There are three ways you can generate flatbuffers, with each suited to a different scenario:
* `ttir-builder` - easiest for creating flatbuffers from custom MLIR code. It is ideal for quick experiments or small test cases. 
* **Compiler (`ttmlir-opt` + `ttmlir-translate`)** - for full control over the pipeline, system descriptors, and backend transformations. It is best for when you are developing or testing compiler passes.
* `llvm-lit` - used for existing test cases or automated regression tests. It generates flatbuffers directly from `.mlir` test files under `test/ttmlir/Silicon`.

### Using ttir-builder to create a flatbuffer
`ttir-builder` is a tool for creating Tenstorrent Intermediate Representation (TTIR) ops, converting them into MLIR modules, running passes to lower modules into backends, and translating to flatbuffers. See [documentation](./ttir-builder/ttir-builder.md) for further instructions.

### Using the compiler to create a flatbuffer
The compiler supports a pass to load a system descriptor to compile against. You can feed this pass into `ttmlir-opt`. (`ttmlir-opt` is a command-line tool for transforming and analyzing MLIR modules that are specific to tt-mlir. It takes an MLIR file as input, applies optimization/transformation passes to it, and produces a modified MLIR file.)

1. Build [ttmlir](./getting-started.md)

2. Generate a ttsys file from the system you want to compile for using `ttrt`. This will create a `system_desc.ttsys` file under `ttrt-artifacts` folder.
```bash
ttrt query --save-artifacts
```

3. Use the `ttmlir-opt` tool in the compiler to feed the system descriptor. See the [`ttmlir-opt`](./ttmlir-opt.md) documentation for more information on how to generate .mlir files.
```bash
./build/bin/ttmlir-opt --ttcore-register-device="system-desc-path=/path/to/system_desc.ttsys" --ttir-to-ttnn-backend-pipeline test/ttmlir/Dialect/TTNN/simple_subtract.mlir -o ttnn.mlir
or (pipe path directly into ttir-to-ttnn-backend-pipeline)
./build/bin/ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=/path/to/system_desc.ttsys" test/ttmlir/Dialect/TTNN/simple_subtract_to_add.mlir -o ttnn.mlir
```

4. Use `ttmlir-translate` tool in compiler to generate the flatbuffer executable. See the [`ttmlir-translate`](./ttmlir-translate.md) documentation for more information on how to generate flatbuffer files.
```bash
./build/bin/ttmlir-translate --ttnn-to-flatbuffer ttnn.mlir -o out.ttnn
```

5. Run your test cases using `ttrt`
```bash
ttrt run /path/to/out.ttnn
```

### Using llvm-lit to generate a flatbuffer
There are already existing `.mlir` test cases under `test/ttmlir/Silicon`. You can use the `llvm-lit` tool to generate the corresponding ttnn and ttm files.

1. Build [ttmlir](./getting-started.md)

2. Generate a ttsys file from the system you want to compile for using `ttrt`. This will create a `system_desc.ttsys` file in the `ttrt-artifacts` folder.
```bash
ttrt query --save-artifacts
```

3. Export this file in your environment using `export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys`. When `llvm-lit` is run, it will query this variable and generate the ttnn and ttm files using this system. Optionally, you can also provide this manually when running `llvm-lit`.

4. Generate your test cases. This will generate all your ttnn and ttm files under `build/test/ttmlir/Silicon`. ttnn files have a `.ttnn` file extension and ttmetal files have a `.ttm` extension.
```bash
cmake --build build -- check-ttmlir
```

5. (Optional) If you created your own .mlir file (or a directory of them) and want to generate the corresponding .ttnn and .ttm outputs, you can run `llvm-lit` directly on that file or directory to produce the flatbuffer executables. You will have to make sure you add in the correct `llvm-lit` configs into your .mlir file. See the section on adding `llvm-lit` config options inside a .mlir file to create flatbuffer binaries for more info. You must also place your .mlir test in the test/ttmlir/Silicon directory and run `llvm-lit` from (or pointed at) your build directory.

```bash
llvm-lit -v ./build/test/ttmlir/Silicon
or
SYSTEM_DESC_PATH=/path/to/system_desc.ttsys llvm-lit -v ./build/test/ttmlir/Silicon
```

6. Run your test cases using `ttrt`:

```bash
ttrt run /path/to/test.ttnn
ttrt run /path/to/dir/of/flatbuffers
```

### Adding llvm-lit config options inside a .mlir file to create flatbuffer binaries
Inside of your .mlir file, you can add certain config options that `llvm-lit` uses when running against that test case. For the purpose of generating flatbuffer executables, you can add `--ttcore-register-device="system-desc-path=%system_desc_path%"` which will tell `llvm-lit` to parse the system desc found from the environment flag set by `export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys`. You can also paste a custom path to a system desc file as well.

```bash
// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-layout --convert-ttir-to-ttnn %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
```

## Adding new mlir test cases
You can copy your .mlir test file (with the appropriate `llvm-lit` config options for generating flatbuffer binaries) into `test/ttmlir/Silicon`. Then run `llvm-lit` from the build directory on that file to generate the `.ttnn` and `.ttm` executables that you can run.

## Versioning
`ttrt` and flatbuffers are version-sensitive. Flatbuffers must be generated with the same major/minor version as `ttrt`. Major and minor versions are manually set using github tags when releases are made. Patch versioning is the number of commits from the last major/minor tag.

```bash
vmajor.minor.patch
```

The flag `--ignore-version` can be used to bypass versioning checks. Use at your own risk; it can cause unpredictable errors.

## ttrt command line commands
This section shows you the basic commands available from the ttrt CLI (command line interface). They can be viewed by typing `ttrt --help`. 

* `ttrt query` - Query for information about the current system.
* `ttrt read` - Read and display sections from a TTNN flatbuffer binary.
* `ttrt run` - Execute a TTNN flatbuffer binary on the device. 
* `ttrt perf` - Run a performance trace and collect performance data.
* `ttrt check` - Validate a TTNN flatbuffer binary against a system descriptor file. 
* `ttrt emitpy` - Run EmitPy Dylib tests and optionally compare outputs to TTNN. 
* `ttrt emitc` - Run EmitC Dylib tests and optionally compare outputs to flatbuffer outputs.
* `ttrt -h` / `ttrt --help` - Show this help message and exit. 
* `ttrt -V` / `ttrt --version` - Show the program's version number and exit. 
* `ttrt --gdb` - Launch ttrt with GNU debugger, a standard command-line debugger for programs written in C, C++, and other compiled languages. 

### Using the ttrt CLI
You can retrieve more information about any `ttrt` command by typing `ttrt -command- --help` (for example, `ttrt read --help`). All artifacts are saved in the `ttrt-artifacts` folder under `TT_MLIR_HOME` environment variable. By default, all logging is printed to the terminal. You can specify a log file to dump output to.

### query 
Query for information about the current system. It differs from the read command in that it focuses on system-level information. It does not look inside a compiled model. 

Available `query` commands include:

* `ttrt query --help` - Pulls up a list of all available commands for `ttrt query`. 

#### Artifact management
* `ttrt query --clean-artifacts` - Remove artifacts from previous runs.
* `ttrt query --save-artifacts` - Save all artifacts generated during a run.
* `ttrt query --artifact-dir ARTIFACT_DIR` - Save artifacts to a specified directory.

#### Logging
* `ttrt query --log-file LOG_FILE` - Write `ttrt` output to the specified log file.
* `ttrt query --quiet` - Suppress system desc from being printed.

#### Execution control
* `ttrt query --disable-eth-dispatch` - Disable putting dispatch on ethernet cores — place it on worker cores instead.

#### Result output
* `ttrt query --result-file RESULT_FILE` - Save query results to a file.

### read
Read sections of a binary file. You must specify a valid TTNN file generated by TT-Forge/tt-mlir. Some commands also accept directories of TTNN/flatbuffer files. In the list of commands list provided, the following conventions are used:
* `out.ttnn` as a placeholder for a compiled binary file
* `/dir/of/flatbuffers` as a placeholder for flatbuffer files. When a folder is specified, all files in the folder are reviewed. 
* `/path/to/some/dir` is a placeholder for the location where you want to place saved artifacts.

Available `read` commands include:

* `ttrt read --help` - Pulls up a list of all available commands for `ttrt read`.

#### Read single sections
Read specific sections of a flatbuffer binary. Replace `out.ttnn` with your binary file.

* `ttrt read --section version out.ttnn` - Display the version information embedded in the flatbuffer (TTNN version, schema version, etc.).
* `ttrt read --section system_desc out.ttnn` - Display the system descriptor stored inside the flatbuffer (the hardware target it was built for).
* `ttrt read --section mlir out.ttnn` - Display the embedded MLIR module contained in the flatbuffer — this is the lowered IR the compiler generated.
* `ttrt read --section inputs out.ttnn` - Show metadata about input tensors (shapes, dtypes, layout).
* `ttrt read --section outputs out.ttnn` - Show metadata about output tensors the program produces.
* `ttrt read --section op_stats out.ttnn` - Show operation statistics (counts, op types, memory usage metadata if present).
* `ttrt read --section mesh_shape out.ttnn` - Display the device mesh shape the program expects (e.g., 1×1, 1×8, 2×4).

#### Read all sections
Read all sections of a binary at once. Optionally clean or save artifacts.

* `ttrt read --section all out.ttnn --clean-artifacts` - Display all sections, and remove artifacts from previous runs before writing new ones.
* `ttrt read --section all out.ttnn --save-artifacts` - Display all sections and save any artifacts generated by read (e.g., JSON dumps of the sections).
* `ttrt read --section all /dir/of/flatbuffers` - Read all flatbuffers within a directory, one by one, and print all sections for each of them.

#### Read system descriptor files
Read a system descriptor directly instead of from a flatbuffer.

* `ttrt read system_desc.ttsys` - Load and display a system descriptor file directly (outside of a flatbuffer).
* `ttrt read --section system_desc system_desc.ttsys` - Same as above, but explicitly specifies the system_desc section.

#### Logging and saving output
Customize where output is logged or saved, or export results for programmatic inspection.

* `ttrt read system_desc.ttsys --log-file ttrt.log` - Save all logging output to `ttrt.log` instead of printing it to terminal.
* `ttrt read out.ttnn --save-artifacts --artifact-dir /path/to/some/dir` - Save the read results (e.g., MLIR dump, input/output metadata) into a custom artifact directory.
* `ttrt read out.ttnn --result-file result.json` - Write the extracted sections to a JSON file for programmatic inspection.

### run
Use `ttrt run` to execute a TTNN flatbuffer binary on device hardware. This command can also perform golden-checking, debugging, tensor inspection, repeated execution, and kernel dumping.

* `ttrt run --help` - Pulls up a list of all available commands for `ttrt run`.

#### Initialization and run control 
Control artifacts, logging, and basic execution parameters for run.

* `ttrt run --clean-artifacts` - Clean all artifacts from previous runs.
* `ttrt run --save-artifacts` - Save all artifacts generated during execution.
* `ttrt run --log-file LOG_FILE` - Write logs to a specified file instead of stdout.
* `ttrt run --artifact-dir ARTIFACT_DIR` - Override the default artifact directory.
* `ttrt run --program-index {all,0,1,2,3,4}` - Select which program inside the flatbuffer to run.
* `ttrt run --loops LOOPS` - Run the program multiple times in a loop.

#### Tensor initialization
Set up input tensors before running, optionally using deterministic patterns. 

* `ttrt run --init {arange,ones,randn,zeros}` - Initialize input tensors using the specified pattern (disables golden).
* `ttrt run --identity` - Check that outputs match identity output tensors.
* `ttrt run --non-zero` - Validate that the output tensors contain non-zero values.
* `ttrt run --seed SEED` - Seed for random number generation used in tensor initialization.

#### Golden checking and validation
Compare program outputs against expected (golden) results with configurable tolerances. For more details on how the run command handles golden checks, see the section [Golden checks](#golden-checks).

* `ttrt run --rtol RTOL` - Relative tolerance for identity and PCC golden checks.
* `ttrt run --atol ATOL` - Absolute tolerance for identity and PCC golden checks.
* `ttrt run --rtol-allclose RTOL_ALLCLOSE` - rtol for allclose-style golden check.
* `ttrt run --atol-allclose ATOL_ALLCLOSE` - atol for allclose-style golden check.
* `ttrt run --pcc PCC` - Minimum Pearson correlation for golden check.
* `ttrt run --golden-diff-topk GOLDEN_DIFF_TOPK` - Print the top-N differences between golden and output tensors.
* `ttrt run --disable-golden` - Disable golden comparisons entirely.
* `ttrt run --save-golden-tensors` - Save golden and output tensors for inspection.

>**NOTE:** Input initialization flags like `--init` automatically disable golden comparison.

#### Kernel and device control
Configure kernel dumping, loading, device execution behavior, and hardware workarounds. 

* `ttrt run --dump-kernels` - Dump kernels to disk as they are executed.
* `ttrt run --load-kernels` - Load previously dumped kernels instead of using flatbuffer.
* `ttrt run --use-loc-for-kernel-name` - Use the operation location to derive kernel filenames when dumping.
* `ttrt run --kernel-source-dir KERNEL_SOURCE_DIR` - Directory to save kernels to.
* `ttrt run --disable-device-address-validation` - Validate device addresses are within legal ranges. (Used when you are confident that device addresses are valid; skipping validation can slightly improve performance.)
* `ttrt run --blocking-cq` - Enable blocking completion queue mode for device execution. (Used when your device execution seems asynchronous or results appear out of order.)
* `ttrt run --disable-swap-binary-operands` - Disable workaround for swapping binary operands. (Used if your binary operands are known to be in correct order and you want to skip the workaround.)
* `ttrt run --disable-read-update-index-for-kv-cache` - Disable read update index workaround for KV cache. (Only used on hardware where the KV cache read-update workaround triggers incorrect behavior.)
* `ttrt run --disable-trace-implicit-from-device` - Disable implicit host-device tracing. (Used to skip implicit host-device tracing to reduce log overhead if you do not need detailed tracing.)
* `ttrt run --disable-blackhole-workarounds` - Disable blackhole workarounds on special cores. (You only need this when running on hardware with blackhole cores and you encounter dispatch errors.)
* `ttrt run --disable-eth-dispatch` - Disable putting dispatch on Ethernet cores; necessary on blackhole cores. (Used if you are running on blackhole cores connected via Ethernet and dispatch must be constrained.)

#### Debugging and inspection
Options for printingtensors, debugging, and memory inspection. 

* `ttrt run --print-input-output-tensors` - Print input and output tensors during execution.
* `ttrt run --debugger` - Launch step debugger after each operation. (Used when you want to step through operations interactively; mainly for debugging kernels or tensor computation.)
* `ttrt run --memory` - Dump memory reports after each operation (use with `--save-artifacts`). (Used when you want detailed memory usage reports for each op (requires `--save-artifacts`).)
* `ttrt run --check-memory-leak` - Check for memory leaks (use with `--memory`). (Used after `--memory` when you want to detect potential memory leaks in runtime execution.)

#### Result output
Save run results for programmatic use or later inspection. For more details about the output file, see the section [Run results](#run-results).

* `ttrt run --result-file RESULT_FILE` - Save run results to a specified file in JSON format.

#### Advanced runtime features
Enable performance tracing, benchmarking, caching, tensor scheduling, and fabric configuration. 

* `ttrt run --dirty-tensor-schedule DIRTY_TENSOR_SCHEDULE` - Schedule tensor dirtying, format: `index:iterations,...` (e.g., `0:1,2:3`). (Only used when testing tensor scheduling effects or specific performance experiments.)
* `ttrt run --enable-program-cache` - Enable program caching in TTNN runtime. (Improves runtime performance when repeatedly running the same program.)
* `ttrt run --trace-region-size TRACE_REGION_SIZE` - Device trace region size. (Used for tuning trace granularity for performance profiling.)
* `ttrt run --dump-device-rate DUMP_DEVICE_RATE` - Rate at which to flush device performance information. (Used when you need periodic device performance reports during long-running programs.)
* `ttrt run --enable-perf-trace` - Enable performance tracing. (Used when detailed performance profiling is required.)
* `ttrt run --benchmark` - Run benchmark mode (warmup + end-to-end time measurements; enables program cache). (Used to measure end-to-end performance or compare runtimes.)
* `ttrt run --ones-density ONES_DENSITY` - Random tensor ones/zeros density (1=100%, 2=50%, 3=33%, etc.). (Only relevant when initializing tensors with patterns that require partial densities.)
* `ttrt run --fabric-config FABRIC_CONFIG` - Set fabric topology: disabled, fabric_1d, fabric_1d_ring, fabric_2d, fabric_2d_torus, fabric_2d_dynamic, custom. (Used when your device topology differs from the default or you need custom fabric configuration.)
* `ttrt run --disable-ttrt-callbacks` - Disable TTNN runtime callbacks. (Used when callbacks interfere with your testing or performance measurements.)
* `ttrt run --ignore-version` - Ignore Major/Minor/Patch mismatch between flatbuffer and TTRT runtime. (Only used when the flatbuffer and runtime mismatch is minor and safe to ignore.)

>**NOTE:** EmitC is an internal representation used by TT-Forge during compilation. If you are just running TTNN binaries with `ttnn-standalone`, you do not typically need to interact with EmitC. Developers working on the compiler internals can refer to the [EmitC testing page](emitc-testing.md). 

#### Key concepts for the run command
This section provides additional details about commands where more information may be needed to fully understand them. 

##### Run results
The `run` command produces a `run_results.json` file that contains information about the run including any errors that were thrown and location of other saved run data.


```json
{
[
  {
    "file_path": "ttnn/test_tan[f32-shape0]_ttnn.mlir.ttnn",
    "result": "pass",
    "exception": "",
    "log_file": "ttrt.log",
    "artifacts": "/home/$USER/tt-mlir/ttrt-artifacts",
    "program_index": "all",
    "program_results": {
      "program_index_0": {
        "loop_0": {
          "total_duration_ns": 3269341588,
          "total_ttnn_api_duration_ns": null,
          "total_device_kernel_duration_ns": null
        }
      }
    }
  }
]
```

##### Golden checks
Golden checks are used to verify runtime op accuracy. They are run by default during the golden callback unless the flag `--disable-golden` is used. If flag `--save-artifacts` is used, a golden results report will be saved under the artifacts directory.

<details>

```json
{
    "loc(\"/home/$USER/tt-mlir/test/python/golden/test_ttir_ops.py:74:id(0)\")": {
        "expected_pcc": 0.99,
        "actual_pcc": 0.0015917614829425491,
        "atol": 1e-08,
        "rtol": 1e-05,
        "allclose": false,
        "max": 8529.765625,
        "mean_absolute_error": 6.644593238830566,
        "root_mean_square_error": 100.30211639404297,
        "cosine_similarity": 0.0016297339461743832
    }
}
```

</details>

##### Memory
Memory callback functions are run when the `--memory` flag is used. A memory report will be written under the artifacts directory that contains information on op memory usage.

Example report: 

<details>

```json
{
    "0": {
        "loc": "loc(\"/home/$USER/tt-mlir/test/python/golden/test_ttir_ops.py:74:id(0)\")",
        "debug_str": "%0 = \"ttnn.tan\"(%arg0) : (tensor<128x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<128x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>>> loc(\"/home/$USER/tt-mlir/test/python/golden/test_ttir_ops.py:74:id(0)\")",
        "dram": {
            "num_banks": 12,
            "total_bytes_per_bank": 1071181792,
            "total_bytes_allocated_per_bank": 16384,
            "total_bytes_free_per_bank": 1071167456,
            "largest_contiguous_bytes_free_per_bank": 1071165408,
            "block_table": [
                {
                    "allocated": "yes",
                    "nextID": "1",
                    "prevID": "-1",
                    "size": "8192",
                    "address": "0",
                    "blockID": "0"
                },
                {
                    "allocated": "yes",
                    "nextID": "3",
                    "prevID": "0",
                    "size": "8192",
                    "address": "8192",
                    "blockID": "1"
                },
                {
                    "allocated": "no",
                    "nextID": "-1",
                    "prevID": "1",
                    "size": "1071165408",
                    "address": "16384",
                    "blockID": "3"
                }
            ]
        },
        "l1": {
            "num_banks": 64,
            "total_bytes_per_bank": 1369120,
            "total_bytes_allocated_per_bank": 0,
            "total_bytes_free_per_bank": 1369120,
            "largest_contiguous_bytes_free_per_bank": 1369120,
            "block_table": [
                {
                    "allocated": "no",
                    "nextID": "-1",
                    "prevID": "-1",
                    "size": "1369120",
                    "address": "0",
                    "blockID": "0"
                }
            ]
        },
        "l1_small": {
            "num_banks": 64,
            "total_bytes_per_bank": 32768,
            "total_bytes_allocated_per_bank": 0,
            "total_bytes_free_per_bank": 32768,
            "largest_contiguous_bytes_free_per_bank": 32768,
            "block_table": [
                {
                    "allocated": "no",
                    "nextID": "-1",
                    "prevID": "-1",
                    "size": "32768",
                    "address": "0",
                    "blockID": "0"
                }
            ]
        },
        "trace": {
            "num_banks": 12,
            "total_bytes_per_bank": 0,
            "total_bytes_allocated_per_bank": 0,
            "total_bytes_free_per_bank": 0,
            "largest_contiguous_bytes_free_per_bank": 0,
            "block_table": [
                {
                    "allocated": "no",
                    "nextID": "-1",
                    "prevID": "-1",
                    "size": "0",
                    "address": "0",
                    "blockID": "0"
                }
            ]
        }
    }
}
```

</details>

#### Debugger
Enabling the `--debugger` flag sets a [pdb trace](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://docs.python.org/3/library/pdb.html&ved=2ahUKEwjT6Znv9oWOAxXe48kDHVs2KwAQFnoECBcQAQ&usg=AOvVaw3vJ9FXJKiMDkCwRHDUYrsr) to run after each op during the callback hook.

### perf
Run performance mode of a binary file or a directory of binary files.

>**NOTE:** You can collect host only related performance data via `--host-only` flag. By default, host and device side performance data are both collected. 

If the saving artifacts flag is provided, perf mode will dump the following files in the artifacts directory:
* `profile_log_device.csv` - Dump of all device side profiled results. 
* `tracy_ops_data.csv` - Op data results dumped in a readable format. 
* `tracy_ops_times.csv` - Op time results dumped in a readable format. 
* `tracy_profile_log_host.tracy` - tracy profiled results file, this file can be fed into the tracy GUI. 

Using the saving artifacts flag also dumps `ops_perf_results.csv`, which shows compiled op performance results. This file contains a CSV row for each operation executed, including: 
* Operation metadata (type, attributes, device, call counts)
* Host and device timing info
* Kernel execution duration per core
* Input/output tensor metadata (shapes, layout, datatype)
* Kernel source references and hashes 
* Performance metrics (compute times, bandwidth, FPU utilization)
* Program/run metadata (loop number, program index, runtime flags)

You can view a [sample row from this file here.](sample_ops_perf_results_row.md) 

Available `perf` commands include: 

`ttrt perf --help` - Pulls up a list of all available commands for `ttrt perf`. 

#### Initialization and run control
Controls which programs are run, how many times, and where artifacts are stored.
* `ttrt perf --clean-artifacts` - Remove artifacts from previous runs before collecting new performance data.
* `ttrt perf --log-file LOG_FILE` - Write ttrt output to a specified log file instead of stdout.
* `ttrt perf --artifact-dir ARTIFACT_DIR` - Save artifacts to a specified directory.
* `ttrt perf --program-index {all,0,1,2,3,4}` - Select which program inside the flatbuffer to profile.
* `ttrt perf --loops LOOPS` - Run the performance trace multiple times.

#### Tracing and measurement control
Options for controlling device tracing, region sizes, and performance measurement behavior.
* `ttrt perf --host-only` - Collect performance trace on host only, without involving device cores.
* `ttrt perf --port PORT` - Port to run Tracy client-server application.
* `ttrt perf --trace-region-size TRACE_REGION_SIZE` - Device trace region size.
* `ttrt perf --dump-device-rate DUMP_DEVICE_RATE` - Rate at which device performance info is flushed.
* `ttrt perf --benchmark` - Enable benchmark mode with warmup and end-to-end timing (automatically enables program cache).

#### Runtime and golden options
Settings that affect runtime execution, golden tensor comparison, and dispatch behavior.
* `ttrt perf --disable-golden` - Disable golden comparison for intermediate and output tensors.
* `ttrt perf --disable-eth-dispatch` - Disable dispatch on Ethernet cores; use worker cores instead.
* `ttrt perf --ignore-version` - Ignore Major/Minor/Patch mismatch between flatbuffer and TTRT runtime.
* `ttrt perf --enable-program-cache` - Enable program cache in TTNN runtime.
* `ttrt perf --disable-ttrt-callbacks` - Disable TTNN runtime callbacks.
* `ttrt perf --emitc` - Toggle EmitC testing during performance runs.

#### Inspection and filtering
Options for inspecting memory usage and filtering performance data.
* `ttrt perf --memory` - Dump memory reports after every operation execution.
* `ttrt perf --filter FILTER` - Comma-separated list of operation types to exclude from performance results (e.g., `const_eval,input_layout_conversion`).

```bash
ops_perf_results.csv : compiled op performance results
```

You can view a [sample row from the ops_perf_results.csv file here.](sample_ops_perf_results_row.md)

```bash
profile_log_device.csv : dump of all device side profiled results
tracy_ops_data.csv : op data results dumped in a readable format
tracy_ops_times.csv : op time results dumped in a readable format
tracy_profile_log_host.tracy : tracy profiled results file, this file can be fed into the tracy GUI
```

### check
Check a binary file or a directory of binary files against a system descriptor (by default, uses the host machine).

You can specify a system descriptor with `--system-desc` to check against a particular target. By default, the host machine’s system descriptor is used.

When using the saving artifacts flags, `check` mode can generate outputs in the artifacts directory, including intermediate artifacts and structured results if a result file is specified.

Available `check` commands include:

* `ttrt check --help` - Show a list of available commands and options.  

#### Initialization and artifact control
Manage artifacts and logging behavior for the check run.  

* `ttrt check --clean-artifacts` - Remove artifacts from previous runs.  
* `ttrt check --save-artifacts` - Save all artifacts generated during execution.  
* `ttrt check --log-file LOG_FILE` - Write logs to a specified file instead of printing to stdout.  
* `ttrt check --artifact-dir ARTIFACT_DIR` - Override the default artifact directory.  

#### System and binary specification
Define the binary to check and optionally the system descriptor to validate against.  

* `ttrt check binary` - Positional argument specifying the flatbuffer binary file to check.  
* `ttrt check --system-desc SYSTEM_DESC` - System descriptor to check against (optional).  

#### Result output
Control how and where results are saved.  

* `ttrt check --result-file RESULT_FILE` - Save the check results to a JSON file or similar for programmatic inspection.  

### emitpy
Run Python-emitting mode for a flatbuffer binary (or directory of binaries) using TTRT.

`emitpy` allows executing flatbuffer binaries while optionally comparing outputs, dumping performance info, and generating Python artifacts for inspection. You can also enable benchmark or memory reporting modes.

Available `emitpy` commands include:

* `ttrt emitpy --help` - Show a list of available commands and options.  

#### Initialization and artifact control
Manage artifacts and logging behavior for the emitpy run.

* `ttrt emitpy --clean-artifacts` - Remove artifacts from previous runs.  
* `ttrt emitpy --save-artifacts` - Save all artifacts generated during execution.  
* `ttrt emitpy --log-file LOG_FILE` - Write logs to a specified file instead of printing to stdout.  
* `ttrt emitpy --artifact-dir ARTIFACT_DIR` - Override the default artifact directory.  

#### Program and binary control
Specify which program(s) to run and which flatbuffer binary or directory to use.

* `ttrt emitpy dylib` - Positional argument specifying the flatbuffer binary file to run.  
* `ttrt emitpy --program-index {all,0,1,2,3,4}` - Select the program inside the flatbuffer to execute.  
* `ttrt emitpy --flatbuffer FLATBUFFER` - Provide a file or directory path for flatbuffer binaries to compare outputs against.  

#### Execution and comparison options
Control runtime behaviors and golden comparisons.

* `ttrt emitpy --disable-golden` - Disable comparison with golden outputs for intermediate and final tensors.  
* `ttrt emitpy --memory` - Dump memory reports after every operation execution.  
* `ttrt emitpy --disable-eth-dispatch` - Disable putting dispatch on ethernet cores; place it on worker cores instead.  
* `ttrt emitpy --ignore-version` - Ignore major/minor/patch checks between flatbuffer and TTRT runtime.  
* `ttrt emitpy --enable-program-cache` - Enable program caching in TTNN runtime.  
* `ttrt emitpy --dump-device-rate DUMP_DEVICE_RATE` - Rate at which device perf information is flushed.  
* `ttrt emitpy --benchmark` - Enable benchmark mode with warmup and end-to-end time measurements (automatically enables program cache).  
* `ttrt emitpy --disable-ttrt-callbacks` - Disable TTRT callbacks during execution.  
* `ttrt emitpy --print-input-output-tensors` - Print the input and output tensors for inspection.  

#### Loop and result control
Configure loops and where results are stored.

* `ttrt emitpy --loops LOOPS` - Number of loops to execute for benchmarking.  
* `ttrt emitpy --result-file RESULT_FILE` - Save execution results to a file for later inspection.  

#### emitpy results
The `emitpy` command produces a `emitpy_results.json` file that records information about the run, including any errors encountered and the location of other saved artifacts.

The JSON contains entries for each binary or program executed, including:

* `file_path` - Path to the generated Python file for the binary/program.
* `result` - Indicates if the run passed or failed.
* `exception` - Any exception raised during execution.
* `log_file` - Path to the log file for this run.
* `artifacts` - Directory where artifacts were saved.
* `program_index` - Index of the program that was run (`all` if all programs were executed).

An example entry looks like:

```bash
[
  {
    "file_path": "ttir-builder-artifacts/emitpy/test_binary_ops[add-emitpy-f32-128x128]_ttnn.mlir.py",
    "result": "pass",
    "exception": "",
    "log_file": "ttrt.log",
    "artifacts": "/home/$USER/tt-mlir/ttrt-artifacts",
    "program_index": "all"
  }
]
```

### emitc
The `emitc` command runs `.so` files generated by the TTMLIR C backend. These shared libraries can be executed and optionally compared against golden results. You can also provide a flatbuffer file or directory via `--flatbuffer` for output verification. `emitc` is typically used to validate and debug the C backend outputs and can record logs, traces, and artifacts for deeper inspection.

Like `emitpy`, this command supports optional golden comparison, multi-loop execution, tensor printing, benchmarking mode, and saving intermediate artifacts.

Available `emitc` commands include:

* `ttrt emitc --help` — Show all available commands and options.

#### Initialization and artifact management
Control logging, cleanup, and artifact destinations.

* `ttrt emitc --clean-artifacts` — Remove artifacts from previous runs.  
* `ttrt emitc --save-artifacts` — Save all generated artifacts during execution.  
* `ttrt emitc --log-file LOG_FILE` — Write logs to a specified file.  
* `ttrt emitc --artifact-dir ARTIFACT_DIR` — Specify where artifacts should be saved.  

#### Program selection and execution control
Choose which program to execute and how many iterations to run.

* `ttrt emitc --program-index {all,0,1,2,3,4}` — Run a specific program in the flatbuffer (or all).  
* `ttrt emitc --loops LOOPS` — Number of execution loops.  
* `ttrt emitc --print-input-output-tensors` — Print tensors to stdout for debugging.  

#### Golden comparison and versioning
Enable/disable correctness checks and version compatibility enforcement.

* `ttrt emitc --disable-golden` — Disable golden comparison of intermediate/output tensors.  
* `ttrt emitc --flatbuffer FLATBUFFER` — Reference flatbuffer or directory for golden comparison.  
* `ttrt emitc --ignore-version` — Skip Major/Minor/Patch compatibility checks (unsafe).  

#### Runtime and dispatch behavior
Advanced controls for the underlying TTNN runtime.

* `ttrt emitc --disable-eth-dispatch` — Use worker cores instead of Ethernet cores for dispatch.  
* `ttrt emitc --enable-program-cache` — Enable TTNN runtime program caching.  
* `ttrt emitc --disable-ttrt-callbacks` — Disable TTRT runtime callbacks.  
* `ttrt emitc --memory` — Dump memory reports after every op execution.  

#### Performance and tracing options
Enable device tracing, benchmark mode, and adjustable flush rate.

* `ttrt emitc --trace-region-size TRACE_REGION_SIZE` — Size of device trace buffer.  
* `ttrt emitc --dump-device-rate DUMP_DEVICE_RATE` — Rate at which device performance info is flushed.  
* `ttrt emitc --benchmark` — Enable warmup + end-to-end benchmark mode (also enables program cache).  

#### Positional arguments
* `dylib` — The flatbuffer binary (`.ttnn`) file to execute.

#### emitc results
The `emitc` command produces an `emitc_results.json` file that records information about the run, including any errors encountered and the location of other saved artifacts.

The JSON contains entries for each binary or program executed, including:

* `file_path` - Path to the generated C file for the binary/program.
* `result` - Indicates if the run passed or failed.
* `exception` - Any exception raised during execution.
* `log_file` - Path to the log file for this run.
* `artifacts` - Directory where artifacts were saved.
* `program_index` - Index of the program that was run (`all` if all programs were executed).

An example entry looks like:

```bash
[
  {
    "file_path": "ttir-builder-artifacts/emitc/test_binary_ops[add-emitc-f32-128x128]_ttnn.mlir.c",
    "result": "pass",
    "exception": "",
    "log_file": "ttrt.log",
    "artifacts": "/home/$USER/tt-mlir/ttrt-artifacts",
    "program_index": "all"
  }
]
```
For generating EmitC tests, you can use either `ttnn-standalone` or `ttir-builder`, depending on your workflow.

* **`ttnn-standalone`** produces individual tests for the TTMLIR C backend. It can generate `.mlir` programs, compile them to `.c` or `.so` files, and execute them, making it suitable for quick validation of single kernels or small programs.

* **`ttir-builder`** is a more comprehensive tool for generating and running EmitC tests as part of larger test suites or CI pipelines. It handles multiple binaries, organizes artifacts, and produces structured outputs such as `emitc_results.json`.

For more details on using these tools:

* See [EmitC testing documentation](./emitc-testing.md) for `ttnn-standalone`.
* See [ttir-builder documentation](./builder/ttir-builder.md) for `ttir-builder`.


For info on generating EmitC tests through `ttnn-standalone`, see [EmitC testing documentation](./emitc-testing.md). 

For info on generating EmitC tests through `ttir-builder`, see [ttir-builder documentation](./builder/ttir-builder.md).

### gdb
You can relaunch `ttrt` inside of the GNU debugger (gdb) which can be useful for debugging C++ runtime components.

```bash
ttrt --gdb run ...
ttrt --gdb perf ...
```

## FAQ

### Troubleshooting your tt-mlir build
If you have difficulty building tt-mlir so that ttrt runs with it, here are some common issues you may encounter: 

#### Fix a broken or partial build by cleaning the build directory
If you find yourself needing to troubleshoot a failed build, it often helps to remove the build directory and recreate it:

```bash
rm -rf build
mkdir build
```

#### Fix missing third-party files / corrupt submodule
If you see failures, they are likely to occur for `third_party/tt-metal` or `third_party/tracy`. The fix for this issue is:

```bash
git submodule update --init --recursive
``` 

#### Check for missing system utilities in the build system
An example error message if you are missing a system utility looks like:

```bash
pkg-config: No such file or directory
fatal error: capstone.h: No such file or directory
```

You can fix this sometimes just by cleaning the build directory, and/or you can try installing the following:

```bash
sudo apt-get install pkg-config
sudo apt-get install libcapstone-dev
```
### Flatbuffer or system descriptor mismatch 
`ttrt` enforces strict compatibility checks at runtime. You may encounter errors if:

1. The flatbuffer version does not match the `ttrt` runtime version.
2. The system descriptor used to compile the flatbuffer does not match the current system.

To resolve:

* Regenerate the flatbuffer using the current `ttrt` build.
* Ensure the flatbuffer is compiled on the same branch or system configuration as your runtime.
* If targeting a different machine, generate a flatbuffer with that system’s descriptor.

See the "generate a flatbuffer file from compiler" section for details.

### I just want to test and push my commit! What do I do!
Follow these steps (on n150, n300, and llmbox)

1. Build ttmlir (sample instructions - subject to change)
```bash
source env/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DTTMLIR_ENABLE_RUNTIME=ON -DTT_RUNTIME_ENABLE_PERF_TRACE=ON
cmake --build build
```
2. Query system
```bash
ttrt query --save-artifacts
```

3. Export system desc file
```bash
export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys (path dumped in previous command)
```

4. Generate test cases
```bash
cmake --build build -- check-ttmlir
```

5. Run test cases
```bash
ttrt run build/test/ttmlir/Silicon
```

6. (Optional) Run perf test cases
```bash
ttrt perf build/test/ttmlir/Silicon
```

### TTRT yields an ambiguous segmentation fault!
The `ttrt` toolchain has specific behaviors and requirements that can lead to build and runtime issues, particularly when dealing with version mismatches or out-of-sync dependencies.

#### Version mismatch due to local commits
The `ttrt` toolchain verifies whether the current system configuration matches the model’s compilation environment. This verification involves tracking the number of commits since the last synchronization. When local commits are made in your branch, it may trigger a version mismatch between the compiled model and the current environment. This mismatch may not be handled properly by the runtime (`rt`), leading to potential issues.

To resolve issues stemming from these synchronization problems, follow this workflow:

1. **Incremental build**
```bash
# make some changes
# commit
cmake --build build
# note you need to generate system_desc and flatbuffer again once you do this
```

This incremental build should be sufficient. If it does not resolve the error, please file an issue and proceed with the following steps for now.

2. **Clear the existing build and dependencies:**
```bash
rm -rf build third_party/tt-metal
```

This ensures that all previous build artifacts and dependencies are removed, preventing conflicts or stale files from affecting the new build.

3. **Rebuild from scratch:**
After clearing the build directories, rebuild the project from the ground up. This ensures that the build process incorporates all the necessary components without any remnants of previous builds. [Build Instructions](./getting-started.md#building-the-tt-mlir-project)

4. **Switch build configurations:**
If switching from a Debug to a Release build (or vice versa), ensure that you clean the build environment before transitioning. This avoids inconsistencies between build configurations and potential issues with optimization levels or debugging symbols.

5. **Re-acquire the IRD:**
By relinquishing and re-acquiring the IRD, you ensure that the correct toolchain is used for the new build. This step ensures synchronization between the model and the toolchain.

6. **Enable Debug Logging for tt-metal:**
To gain more insight into potential issues, enable debugging by setting the TT_METAL_LOGGER_LEVEL to DEBUG. This will provide detailed logs, which can help in troubleshooting build or runtime issues.

```bash
export TT_METAL_LOGGER_LEVEL=DEBUG
```

