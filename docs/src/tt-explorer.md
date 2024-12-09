# `tt-explorer`

Welcome to the tt-explorer wiki! The Wiki will serve as a source for documentation, examples, and general knowledge related to the TT-MLIR visualization project. The sidebar will provide navigation to relevant pages. If this is your first time hearing about the project, take a look at Project Architecture for an in-depth introduction to the tool and motivations behind it :)

## Quick Start
TT-Explorer is made to be as painless as possible, as such the installation on top of the pre-existing [`tt-mlir`](https://github.com/tenstorrent/tt-mlir) project is as minimal as possible.

1. Build `tt-mlir`, add the `-DTT_EXPLORER_EDITABLE=ON` flag to the cmake build to install the `tt-explorer` package in editable mode.
2. Run `source env/activate` to be in `tt-mlir` virtualenv for the following steps
3. Install the explorer tool by building the `explorer` target using `cmake --build build -- explorer`
4. Run `tt-explorer` in terminal to start tt-explorer instance. (Refer to CLI section in API for specifics)
5. Ensure server has started in `tt-explorer` shell instance (check for message below)
```sh
Starting Model Explorer server at:
http://localhost:8080
```

Visualizer tool for `ttmlir`-powered compiler results. Visualizes from emitted `.mlir` files to display compiled model, attributes, performance results, and provide a platform for human-driven overrides to _gameify_ model tuning.

## TT-Explorer - Project Architecture

TT-Explorer is a tool made to ease the pain of tuning a model and developing on Tenstorrent hardware. It provides a “Human-In-Loop” interface such that the compiler results can be actively tuned and understood by the person compiling the model. To complete this goal, the tool has to be designed such that users of any level of experience are all able to glean useful information from the visualization of the model, and be able to **explore** what the model does.

### Software Architecture

The software will be built around the TT-Forge compiler to provide most of the functionality. [Model Explorer](https://github.com/google-ai-edge/model-explorer) will be used for the visualization functionality and as the main platform upon which TT-Explorer is built on.

Since Model-Explorer is built using Python, the majority of TT-Explorer will be structured in Python, with frequent use of the bindings to C++ provided by TT-MLIR.

The following components will be put together:

![ttExplorerArchitecture](https://github.com/user-attachments/assets/f996af27-8b66-4579-a6d6-ded57cbe89d1)

#### [TT-Forge-FE (Front End)](https://github.com/tenstorrent/tt-forge-fe)

TT-Forge FE is *currently* the primary frontend which uses TVM to transform conventional AI models into the MLIR in the TTIR Dialect.

**Ingests**: AI Model defined in PyTorch, TF, etc…
**Emits**: Rudimentary TTIR Module consisting of Ops from AI Model.

#### [TT-MLIR](https://docs.tenstorrent.com/tt-mlir/overview.html)

TT-MLIR currently defines the out-of-tree MLIR compiler created by Tenstorrent to specifically target TT Hardware as a backend. It comprises a platform of several dialects (TTIR, TTNN, TTMetal) and the passes and transformations to compile a model into an executable that can run on TT hardware. In the scope of TT-Explorer the python bindings will be leveraged.

**Ingests**: TTIR Module, Overrides JSON
**Emits:** Python Bindings to interface with TTIR Module, Overridden TTIR Modules, Flatbuffers

#### [TT-Adapter](https://github.com/vprajapati-tt/tt-adapter)

Model Explorer provides an extension interface where custom adapters can be implemented to visualize from different formats. TT-Adapter is the adapter created for TT-Explorer that parses TTIR Modules using the Python Bindings provided by TT-MLIR to create a graph legible by model-explorer. It also has an extensible REST endpoint that is leveraged to implement functionality, this endpoint acts as the main bridge between the Client and Host side processes.

**Ingests**: TTIR Modules, TT-MLIR Python Bindings, REST API Calls
**Emits**: Model-Explorer Graph, REST API Results

#### [TTRT](https://docs.tenstorrent.com/tt-mlir/ttrt.html)

TT-RT is the runtime library for TT-Forge, which provides an API to run Flatbuffers generated from TT-MLIR. These flatbuffers contain the compiled results of the TTIR module, and TTRT allows us to query and execute them. Particularly, a performance trace can be generated using Tracy, which is fed into model-explorer to visualize the performance of operations in the graph.

**Ingests**: Flatbuffers
**Emits**: Performance Trace, Model Results

#### [Model-Explorer](https://github.com/google-ai-edge/model-explorer)

Model Explorer is the backbone of the client and visualization of these models. It is deceptively placed in the “Client” portion of the diagram, but realistically TT-Explorer will be run on the host, and so will the model-explorer instance. The frontend will be a client of the REST API created by TT-Adapter and will use URLs from the model-explorer server to visualize the models.

**Ingests**: Model Explorer Graph, User-Provided Overrides (UI), Performance Trace
**Emits**: Overrides JSON, Model Visualization

These components all work together to provide the TT-Explorer platform.

### Client-Host Design Paradigm

Since performance traces and execution rely on Silicon machines, there is a push to decouple the execution and MLIR-environment heavy aspects of TT-Explorer onto some host device and have a lightweight client API that uses the REST endpoint provided by TT-Adapter to leverage the host device without having to constantly be on said host. This is very useful for cloud development (as is common Tenstorrent). In doing so, TT-Explorer is a project that can be spun up in either a `tt-mlir` environment, or without one. The lightweight python version of TT-Explorer provides a set of utilities that call upon and visualize models from the host, the host will create the server and serve the API to be consumed.
