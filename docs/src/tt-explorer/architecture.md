# `tt-explorer` - Project Architecture

TT-Explorer is a tool for exploring, visualizing, and executing machine learning models on Tenstorrent hardware. It provides a Human-In-Loop interface that allows developers to:
* Inspect compiler results
* Apply configuration overrides
* Observe the effect of changes on model execution

Through TT-Explorer's interactive visualizations, users can examine model operations, performance traces, and execution results, providing insight into model behavior and supporting optimization efforts.

## Software Architecture

TT-Explorer is structured around the TT-Forge compiler stack, which provides the compilation and transformation capabilities used to generate and visualize models for Tenstorrent hardware. The TT-RT runtime handles the backend execution of compiled models on hardware.

For visualization and interaction, TT-Explorer extends Google's [Model Explorer](https://github.com/google-ai-edge/model-explorer), which serves as the primary UI and graph-rendering framework. Because Model Explorer is implemented in Python, TT-Explorer primarily makes use of Python bindings provided by TT-MLIR for direct access to compiler internals.

The overall system integrates these components:

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

### [TT-Forge-FE (Front End)](https://docs.tenstorrent.com/tt-forge-fe/)

TT-Forge FE is _currently_ the primary frontend which uses TVM to transform conventional AI models into the MLIR in the TTIR Dialect.

**Ingests**: AI Model defined in PyTorch, TF, etc…
**Emits**: Rudimentary TTIR Module consisting of Ops from AI Model.

### [TT-MLIR](../overview.md)

TT-MLIR currently defines the out-of-tree MLIR compiler created by Tenstorrent to specifically target TT Hardware as a backend. It comprises a platform of several dialects (TTIR, TTNN, TTMetal) and the passes and transformations to compile a model into an executable that can run on TT hardware. In the scope of `tt-explorer` the python bindings will be leveraged.

**Ingests**: TTIR Module, Overrides JSON
**Emits:** Python Bindings to interface with TTIR Module, Overridden TTIR Modules, Flatbuffers

### [TT-Adapter](https://github.com/tenstorrent/tt-mlir/tree/main/tools/explorer/tt_adapter)

TT-Adapter is the adapter created for `tt-explorer` that parses TTIR Modules using the Python Bindings provided by TT-MLIR to create a graph legible by model-explorer. It also has an extensible REST endpoint that is leveraged to implement functionality, this endpoint acts as the main bridge between the Client and Host side processes.

It is the piece that connects Model Explorer, through a custom adapters that can visualize TTIR, to the rest of the architecture.

**Ingests**: TTIR Modules, TT-MLIR Python Bindings, REST API Calls
**Emits**: Model-Explorer Graph, REST API Results

### [`ttrt`](../ttrt.md)

TT-RT is the runtime library for TT-Forge, which provides an API to run Flatbuffers generated from TT-MLIR. These flatbuffers contain the compiled results of the TTIR module, and TTRT allows us to query and execute them. Particularly, a performance trace can be generated using Tracy, which is fed into model-explorer to visualize the performance of operations in the graph.

**Ingests**: Flatbuffers
**Emits**: Performance Trace, Model Results

### [Model-Explorer](https://github.com/tenstorrent/model-explorer)

Model Explorer is the backbone of the client and visualization of these models. It is placed in the “Client” portion of the diagram as this is where most interactions with Model Explorer will happen. Due to the "client and server" architecture, `tt-explorer` will be run on the host, and so will the model-explorer instance.

The frontend will be a client of the REST API created by TT-Adapter and will use URLs from the model-explorer server to visualize the models.

Currently TT maintains a fork of model-explorer which has changes to the UI elements for overrides, graph execution, and displaying performance traces.

**Ingests**: Model Explorer Graph, User-Provided Overrides (UI), Performance Trace
**Emits**: Overrides JSON, Model Visualization

These components all work together to provide the `tt-explorer` platform.

## Client-Server Design Paradigm

Since performance traces and execution rely on Silicon machines, there is a push to decouple the execution and MLIR-environment heavy aspects of `tt-explorer` onto some host device with TT hardware on it. Then have a lightweight REST server, provided by TT-Adapter, to leverage the host device without having to constantly be on said host.

And finally connect to a client through a web interface, thus removing the need to run the client and server in the same machine.

This is very useful for cloud development (as is common Tenstorrent). In doing so, `tt-explorer` is a project that can be spun up in either a `tt-mlir` environment, or without one. The lightweight python version of `tt-explorer` provides a set of utilities that call upon and visualize models from the host, the host will start the server and serve the API to be consumed. The client can then be any browser that connects to that server.
