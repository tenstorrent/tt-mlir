# Milestone 1 (v0.1)

## Main Goal \- Visualize & Execute

This will highlight half of the essential work that this tool should be able to do in both visualizing a model and executing it using the current TT-Forge stack. The frontend transformation of a model \-\> TTIR will be done outside of the scope of `tt-explorer` at the moment. For this milestone `tt-explorer` will be able to spin up a host-side and a client-side instance. The tool will be able to ingest TTIR modules to produce a visual result, and be able to execute this module. Ambitiously, the performance traces should be collected back into `tt-explorer` to be displayed.

Tasks:

- [x] ~~Load TTIR Modules and Visualize TTIR-Ops in Model Explorer~~
- [x] ~~Create Extensible Notebook UX allowing for visualization and scripting capabilities~~
- [x] ~~Add functionality to Model Explorer to load from re-compiled TTIR Modules (might be from JSON)~~
- [x] ~~Add functionality to TT-MLIR to execute from Python Bindings~~
- [x] ~~Create REST API skeleton in TT-Adapter~~
- [x] ~~From REST API Call, Invoke python bindings to execute TTIR module using TT-Adapter~~
- [ ] (If possible) Parse Perf Trace Artifact and visualize performance in Model-Explorer (as Node Data)

# Milestone 2 (v0.2)

## Main Goal \- Model Editor

The primary function of `tt-explorer` is to visualize **and edit** the model according to what the user defines as overrides the automatically generated compiler results. This milestone highlights that functionality in `tt-explorer`, focusing around providing UI, TT-MLIR, and `tt-explorer` features that enable the user to edit and tune a model “in-loop” with the TT-Forge compiler.

Tasks:

- [x] ~~Flesh out and test locations ID such that operations can be tracked through the compiler stack.~~
- [x] ~~Use Loc IDs to bind TTIR Ops with Tracy Perf Trace Artifact, and send to Model-Explorer to visualize.~~
- [x] ~~Implement Overrides Functionality into TT-MLIR, tracking based on Loc IDs.~~
- [x] ~~Overhaul UI to enable editing node attributes, use these updated fields to send information back to TT-Explorer via REST API (in the form of an Overrides JSON)~~
- [x] ~~Parse Overrides JSON and apply Overrides over a REST API Call, visualize re-compiled graph now.~~
- [x] ~~Provide REST API endpoint to provide “legal” options attached to Graph JSON.~~

# Milestone 3 (v0.3+)

## Main Goal \- Matured Tool and Extensibility

The focus of this milestone is to transition `tt-explorer` from a prototype tool into a mature visualization and editing tool for “Human-In-Loop” compilation. The tool is now planned to made extensible for other dialects and entry points forecast into TT-MLIR (Jax, StableHLO, etc…) and development of the visualization components of the tool provide feedback to upstream repos like `model-explorer`. Here the focus is on providing extensible interfaces for new UI elements (in supporting multi-chip and beyond), REST API, and Overrides.

Tasks:

- [x] ~~Begin adding new dialects like `.ttm`, `.ttnn` to Model Explorer so that complied results can be inspected and analyzed to optimize at different steps of the compiler.~~
- [x] ~~Add Accuracy/Performance Overlays as Node Data into the Model Explorer graph to visualize execution results~~
- [ ] Enable interaction with `ttnn-visualizer` and other TT Visualizer tools to provide a more detailed view of execution results.
- [ ] Start introducing InterOp with builtin adapters in `model-explorer` to support visualizing models from FE.
- [ ] Use split panes to display graph transformations occuring through compiler, leveraging multiple dialects.
- [ ] *To be defined later, depending on the growth of the MLIR Project*
