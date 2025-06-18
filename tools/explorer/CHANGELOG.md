# TT-Explorer

## v1.0.0

### Highlights

- Model execution ("human in the loop")
- Editable attributes and overrides to models
- Display performance overlays
- Display different data types

### New features

#### Model execution

The largest feature added in the fork of [`model-explorer`](https://github.com/google-ai-edge/model-explorer) is the ability to execute models and load them back into the interface.

The original `model-explorer` was a tool for only visualizing static generated data. Adding a way to execute models from within the tool removes friction and enables users to have a quicker feedback loop on the changes they made and how a model perform, this is the "human in the loop part".

Model execution can be achieved by loading a model and then clicking the "play" button on the top navigation bar. It is part of the interface additions that enable control and a way to update existing models.

#### Choose execution policies

Users can also choose an execution policy to tweak how the model is executed by the backend. This helps fine-tune the resulting execution and comparing performance between different execution policies.

#### Editable attributes

One of the big changes is that the server now returns information about parts of a model that can be changed. this enables tweaking how the execution happens for performance gains.

#### Model overrides

Changes to editable attributes can be downloaded and uploaded to the UI for multiple runs of the same model. This way users can keep a record of their changes and reapply to other graphs for testing.

#### Performance Overlays

Results from the server include an overlay for the execution performance, highlighting potential bottlenecks for better insights on where to optimize the morels.

#### Display different data types

The original `model-explorer` only allowed for plain text to be shown for attributes. There is now an addition to display memory as a progress bar.

#### Generate C++ code

As part of the features exposed through the updated interface, there is an option to ask the backend to output a C code representation of a model that will be displayed in the UI.

### Improvements

N/A

### Bug fixes

- Fixed a bug with arrows for a graph not showing in the correct edge

### Deprecations

N/A

### Known issues

N/A

### Coming soon

- Displaying multiple graphs
- Keeping track of execution steps
- Nested attributes
- Optimization policy documentation

### Documentation

Documentation on TT-Explorer can be found as part of the main [`tt-mlir` docs](https://docs.tenstorrent.com/tt-mlir/tt-explorer.html).
