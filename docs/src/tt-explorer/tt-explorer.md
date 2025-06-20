# `tt-explorer`

Welcome to the tt-explorer wiki! The Wiki will serve as a source for documentation, examples, and general knowledge related to the TT-MLIR visualization project. The sidebar will provide navigation to relevant pages. If this is your first time hearing about the project, take a look at Project Architecture for an in-depth introduction to the tool and motivations behind it :)

## Overview

Visualizer tool for `ttmlir`-powered compiler results. Visualizes from emitted `.mlir` files to display compiled model, attributes, performance results, and provides a platform for human-driven overrides to _gameify_ model tuning.

## Quick Start

TT-Explorer comes packaged as a tool in the `tt-mlir` repo. If you haven't done so yet, please refer to ["Setting up the environment manually"](../getting-started.md#setting-up-the-environment-manually) section from the Getting Started Guide to build the environment manually.

Here is a summary of the steps needed:

1. Clone `tt-mlir` and build the environment
2. Run `source env/activate` to be in `tt-mlir` virtualenv for the following steps
3. Ensure `tt-mlir` is built with atleast these flags:
   - `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`
   - `-DTTMLIR_ENABLE_RUNTIME=ON`
   - `-DTT_RUNTIME_DEBUG=ON`
4. Build `explorer` target in `tt-mlir` using `cmake --build build -- explorer`
5. Run `tt-explorer` in terminal to start tt-explorer instance. (Refer to CLI section in API for specifics)
   - **Note**: `tt-explorer` requires [Pandas](https://pypi.org/project/pandas/) in addition to the `tt-mlir` [System Dependencies](https://docs.tenstorrent.com/tt-mlir/getting-started.html#system-dependencies).
6. Ensure server has started in `tt-explorer` shell instance (check for message below)
   ```
   Starting Model Explorer server at:
   http://localhost:8080
   ```

## Building `tt-explorer`

To build `tt-explorer` you need first to clone and configure the environment for `tt-mlir`. Please refer to the [Getting Started Guide](../getting-started.md).

After building and activating the virtualenv, build `tt-mlir` and ensure the following flags are present, as they are needed for executing models in `tt-explorer` and without them it won't build.

Flags required:

- `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`
- `-DTTMLIR_ENABLE_RUNTIME=ON`
- `-DTT_RUNTIME_DEBUG=ON`

Then build the `explorer` target by running the following command:

```sh
cmake --build build -- explorer
```

After it finishes building, start the `explorer` server by running the following command:

```sh
tt-explorer
```

The server should then start and show a message similar to this:

```
Starting Model Explorer server at:
http://localhost:8080
```

### Running TT-Explorer Tests Locally

TT-Explorer relies on tests that are present in the `tests/` directory as well as tests dynamically created through `llvm-lit`. Below are the steps to replicate the testing procedure seen in CI:

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

## TT-Explorer - Project Architecture

TT-Explorer is a tool made to ease the pain of tuning a model and developing on Tenstorrent hardware. It provides a “Human-In-Loop” interface such that the compiler results can be actively tuned and understood by the person compiling the model. To complete this goal, the tool has to be designed such that users of any level of experience are all able to glean useful information from the visualization of the model, and be able to **explore** what the model does.

### Software Architecture

The software will be built around the TT-Forge compiler to provide most of the functionality. [Model Explorer](https://github.com/google-ai-edge/model-explorer) will be used for the visualization functionality and as the main platform upon which TT-Explorer is built on.

Since Model-Explorer is built using Python, the majority of TT-Explorer will be structured in Python, with frequent use of the bindings to C++ provided by TT-MLIR.

The following components will be put together:

![Diagram for component architecture, transcript below](https://github.com/user-attachments/assets/f996af27-8b66-4579-a6d6-ded57cbe89d1)

<details>
<summary>Diagram transcript</summary>

> Horizontal group labeled "Host side" at the top, with nodes from left to right connected to each other by arrows, and at the end connected to the next group.
> The nodes are:
>
> - "AI Model", with an arrow labeled "Model binary file" to the next node.
> - "TVM", with an arrow labeled "PyBUDA Graph" to the next node.
> - "TT-Forge-FE", with an unlabeled arrow to the next node.
> - "TT-MLIR", with an arrow labeled "MLIR file (.ttir, etc...)" to the "TT-Adapter" node on the next group.
>
> Vertical group at the right side, unlabeled, with nodes from top down connected to each other by arrows, and with some arrows going to the next group.
> The group intersects with the "Host side" group at the "TT-MLIR" node.
> The nodes are:
>
> - "TT-Adapter", with an arrow labeled "Flatbuffer w/ Model Binary" to the next node, an arrow labeled "Overrides JSON (to apply)" to the previous node, and an arrow labeled "HTTPS API (Overrides, MLIR -> JSON, etc...)" to and from the "Model Explorer" node on the next group.
> - "TTRT", with an arrow labeled "HTTPS Server Call" to the next node.
> - "Tracy Results", with an arrow labeled "Performance Trace" to the "UI" node on the next group.
>
> Rectangular group labeled "Client Side", below "Host side" and left of unlabeled group, with interconected nodes by arrows, and with some arrows going to the previous group.
> The nodes are:
>
> - "Model Explorer", with an arrow labeled "HTTPS API (Overrides, MLIR -> JSON, etc...)" to and from the "TT-Adapter" node on the previous group, and an arrow labeled "Overrides (legal configurations)" to and from the next node.
> - "UI", with an arrow labeled "Performance Trace" coming from the "Tracy Results" node on the previous group, an arrow labeled "Overrides (legal configurations)" to and from the previous node, and an unlabeled arrow going to the next node.
> - "Notebook", with an arrow labeled "Scripted Overrides" going to the "Model Explorer" node on this group.

</details>

#### [TT-Forge-FE (Front End)](https://github.com/tenstorrent/tt-forge-fe)

TT-Forge FE is _currently_ the primary frontend which uses TVM to transform conventional AI models into the MLIR in the TTIR Dialect.

**Ingests**: AI Model defined in PyTorch, TF, etc…
**Emits**: Rudimentary TTIR Module consisting of Ops from AI Model.

#### [TT-MLIR](https://docs.tenstorrent.com/tt-mlir/overview.html)

TT-MLIR currently defines the out-of-tree MLIR compiler created by Tenstorrent to specifically target TT Hardware as a backend. It comprises a platform of several dialects (TTIR, TTNN, TTMetal) and the passes and transformations to compile a model into an executable that can run on TT hardware. In the scope of TT-Explorer the python bindings will be leveraged.

**Ingests**: TTIR Module, Overrides JSON
**Emits:** Python Bindings to interface with TTIR Module, Overridden TTIR Modules, Flatbuffers

#### [TT-Adapter](https://github.com/tenstorrent/tt-mlir/tree/main/tools/explorer/tt_adapter)

TT-Adapter is the adapter created for TT-Explorer that parses TTIR Modules using the Python Bindings provided by TT-MLIR to create a graph legible by model-explorer. It also has an extensible REST endpoint that is leveraged to implement functionality, this endpoint acts as the main bridge between the Client and Host side processes.

It is the piece that connects Model Explorer, through a custom adapters that can visualize TTIR, to the rest of the architecture.

**Ingests**: TTIR Modules, TT-MLIR Python Bindings, REST API Calls
**Emits**: Model-Explorer Graph, REST API Results

#### [TTRT](https://docs.tenstorrent.com/tt-mlir/ttrt.html)

TT-RT is the runtime library for TT-Forge, which provides an API to run Flatbuffers generated from TT-MLIR. These flatbuffers contain the compiled results of the TTIR module, and TTRT allows us to query and execute them. Particularly, a performance trace can be generated using Tracy, which is fed into model-explorer to visualize the performance of operations in the graph.

**Ingests**: Flatbuffers
**Emits**: Performance Trace, Model Results

#### [Model-Explorer](https://github.com/tenstorrent/model-explorer)

Model Explorer is the backbone of the client and visualization of these models. It is placed in the “Client” portion of the diagram as this is where most interactions with Model Explorer will happen. Due to the "client and server" architecture, TT-Explorer will be run on the host, and so will the model-explorer instance.

The frontend will be a client of the REST API created by TT-Adapter and will use URLs from the model-explorer server to visualize the models.

Currently TT maintains a fork of model-explorer which has changes to the UI elements for overrides, graph execution, and displaying performance traces.

**Ingests**: Model Explorer Graph, User-Provided Overrides (UI), Performance Trace
**Emits**: Overrides JSON, Model Visualization

These components all work together to provide the TT-Explorer platform.

### Client-Server Design Paradigm

Since performance traces and execution rely on Silicon machines, there is a push to decouple the execution and MLIR-environment heavy aspects of TT-Explorer onto some host device with TT hardware on it. Then have a lightweight REST server, provided by TT-Adapter, to leverage the host device without having to constantly be on said host.

And finally connect to a client through a web interface, thus removing the need to run the client and server in the same machine.

This is very useful for cloud development (as is common Tenstorrent). In doing so, TT-Explorer is a project that can be spun up in either a `tt-mlir` environment, or without one. The lightweight python version of TT-Explorer provides a set of utilities that call upon and visualize models from the host, the host will start the server and serve the API to be consumed. The client can then be any browser that connects to that server.
