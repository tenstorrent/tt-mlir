# `tt-explorer`

Welcome to the `tt-explorer` wiki! The Wiki will serve as a source for documentation, examples, and general knowledge related to the TT-MLIR visualization project. The sidebar will provide navigation to relevant pages. If this is your first time hearing about the project, take a look at Project Architecture for an in-depth introduction to the tool and motivations behind it :)

## Quick Start
`tt-explorer` comes packaged as a tool in the `tt-mlir` repo.

1. Run `source env/activate` to be in `tt-mlir` virtualenv for the following steps
2. Ensure `tt-mlir` is built with atleast these flags:
    - `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON -DTTMLIR_ENABLE_RUNTIME=ON -DTT_RUNTIME_DEBUG=ON`
3. Build `explorer` target in `tt-mlir` using `cmake --build build -- explorer`
5. Run `tt-explorer` in terminal to start `tt-explorer` instance. (Refer to CLI section in API for specifics)
    - Note that `tt-explorer` requires [Pandas](https://pypi.org/project/pandas/) in addition to the `tt-mlir` [System Dependencies](./getting-started.md#system-dependencies).
6. Ensure server has started in `tt-explorer` shell instance (check for message below)
```sh
Starting Model Explorer server at:
http://localhost:8080
```

### Running tt-explorer Tests Locally
`tt-explorer` relies on tests that are present in the `tests/` directory as well as tests dynamically created through `llvm-lit`. Below are the steps to replicate the testing procedure seen in CI:
1. Make sure you're in the `tt-mlir` directory
2. You need to build the explorer target with `cmake --build build -- explorer`
3. Run and save the system descriptor `ttrt query --save-artifacts`
4. Save the system variable `export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys`
5. Run and generate ttnn + MLIR tests: `cmake --build build -- check-ttmlir`
6. Save the relevant test directories:
    - `export TT_EXPLORER_GENERATED_MLIR_TEST_DIRS=$(pwd)/build/test/python/golden/ttnn,$(pwd)/build/test/ttmlir/Silicon/TTNN/n150/perf`
    - `export TT_EXPLORER_GENERATED_TTNN_TEST_DIRS=$(pwd)/build/test/python/golden/ttnn`
7. Run the pytest for `tt-explorer` with `pytest tools/explorer/test/run_tests.py`

or in a concise shell script:
```sh
# Ensure you are present in the tt-mlir directory
source env/activate

# Build Tests
cmake --build build -- explorer
ttrt query --save-artifacts
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
cmake --build build -- check-ttmlir

# Load Tests
export TT_EXPLORER_GENERATED_MLIR_TEST_DIRS=$(pwd)/build/test/python/golden/ttnn,$(pwd)/build/test/ttmlir/Silicon/TTNN/n150/perf
export TT_EXPLORER_GENERATED_TTNN_TEST_DIRS=$(pwd)/build/test/python/golden/ttnn

# Run Tests
pytest tools/explorer/test/run_tests.py
```

Visualizer tool for `ttmlir`-powered compiler results. Visualizes from emitted `.mlir` files to display compiled model, attributes, performance results, and provides a platform for human-driven overrides to _gameify_ model tuning.

## Project Architecture

`tt-explorer` is a tool made to ease the pain of tuning a model and developing on Tenstorrent hardware. It provides a “Human-In-Loop” interface such that the compiler results can be actively tuned and understood by the person compiling the model. To complete this goal, the tool has to be designed such that users of any level of experience are all able to glean useful information from the visualization of the model, and be able to **explore** what the model does.

### Software Architecture

The software will be built around the TT-Forge compiler to provide most of the functionality. [Model Explorer](https://github.com/google-ai-edge/model-explorer) will be used for the visualization functionality and as the main platform upon which `tt-explorer` is built on.

Since Model-Explorer is built using Python, the majority of `tt-explorer` will be structured in Python, with frequent use of the bindings to C++ provided by TT-MLIR.

The following components will be put together:

![ttExplorerArchitecture](https://github.com/user-attachments/assets/f996af27-8b66-4579-a6d6-ded57cbe89d1)

#### [TT-Forge-FE (Front End)](https://github.com/tenstorrent/tt-forge-fe)

TT-Forge FE is *currently* the primary frontend which uses TVM to transform conventional AI models into the MLIR in the TTIR Dialect.

**Ingests**: AI Model defined in PyTorch, TF, etc…
**Emits**: Rudimentary TTIR Module consisting of Ops from AI Model.

#### [TT-MLIR](./overview.md)

TT-MLIR currently defines the out-of-tree MLIR compiler created by Tenstorrent to specifically target TT Hardware as a backend. It comprises a platform of several dialects (TTIR, TTNN, TTMetal) and the passes and transformations to compile a model into an executable that can run on TT hardware. In the scope of `tt-explorer` the python bindings will be leveraged.

**Ingests**: TTIR Module, Overrides JSON
**Emits:** Python Bindings to interface with TTIR Module, Overridden TTIR Modules, Flatbuffers

#### [TT-Adapter](https://github.com/vprajapati-tt/tt-adapter)

Model Explorer provides an extension interface where custom adapters can be implemented to visualize from different formats. TT-Adapter is the adapter created for `tt-explorer` that parses TTIR Modules using the Python Bindings provided by TT-MLIR to create a graph legible by model-explorer. It also has an extensible REST endpoint that is leveraged to implement functionality, this endpoint acts as the main bridge between the Client and Host side processes.

**Ingests**: TTIR Modules, TT-MLIR Python Bindings, REST API Calls
**Emits**: Model-Explorer Graph, REST API Results

#### [`ttrt`](./ttrt.md)

`ttrt` is the runtime library for TT-Forge, which provides an API to run Flatbuffers generated from TT-MLIR. These flatbuffers contain the compiled results of the TTIR module, and `ttrt` allows us to query and execute them. Particularly, a performance trace can be generated using Tracy, which is fed into model-explorer to visualize the performance of operations in the graph.

**Ingests**: Flatbuffers
**Emits**: Performance Trace, Model Results

#### [Model-Explorer](https://github.com/tenstorrent/model-explorer)

Model Explorer is the backbone of the client and visualization of these models. It is deceptively placed in the “Client” portion of the diagram, but realistically `tt-explorer` will be run on the host, and so will the model-explorer instance. The frontend will be a client of the REST API created by TT-Adapter and will use URLs from the model-explorer server to visualize the models. Currently TT maintains a fork of model-explorer which has overriden UI elements for overrides and displaying performance traces.

**Ingests**: Model Explorer Graph, User-Provided Overrides (UI), Performance Trace
**Emits**: Overrides JSON, Model Visualization

These components all work together to provide the `tt-explorer` platform.

### Client-Host Design Paradigm

Since performance traces and execution rely on Silicon machines, there is a push to decouple the execution and MLIR-environment heavy aspects of `tt-explorer` onto some host device and have a lightweight client API that uses the REST endpoint provided by TT-Adapter to leverage the host device without having to constantly be on said host. This is very useful for cloud development (as is common Tenstorrent). In doing so, `tt-explorer` is a project that can be spun up in either a `tt-mlir` environment, or without one. The lightweight python version of `tt-explorer` provides a set of utilities that call upon and visualize models from the host, the host will create the server and serve the API to be consumed.
