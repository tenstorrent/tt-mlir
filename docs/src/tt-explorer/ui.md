# `tt-explorer` UI

For general reference of the UI, refer to the [model-explorer wiki](https://github.com/google-ai-edge/model-explorer/wiki). This section will highlight specific UI elements added to the Tenstorrent fork of model-explorer.

## Model Execution

![Toolbar added by `tt-explorer` fork](../images/tt-explorer/toolbar.png)

In the top right of the screen an additional button has been added to the top bar, it sends the model to the server for execution and updates the visualization once it has been executed. Once the model has executed, _overlays_ are also created. These overlays provide information on how the execution went.

### Performance Overlay

![Example of performance overlays for a graph](../images/tt-explorer/perf-overlay.png)

The performance overlay is generated on **every** execution, it highlights the time it took to execute each node on the graph. This is visualized with a gradient from Yellow -> Red, with Yellow being the lowest time amongst all nodes on the graph, and Red being highest.

### Accuracy Overlay

The accuracy overlay is _only_ generated when executing from a compatible flatbuffer (`.ttnn` file extension with Debug Info). The overlay consists of either Green or Red node overlays. Green if the node passed a "golden" test, Red if not.

The value for the overlay is the actual Pearson Correlation Coefficient (PCC) value with the "golden" tensor subtracted by the expected PCC value. If the number is `< 0` we know it doesn't match the expected PCC, otherwise it is an accurate comparison.

## Advanced Settings

![Toolbar highlighting the "configuration" button](../images/tt-explorer/configure.png)

This menu will open a window with some advanced settings for Model execution.

### Opt. Policy

This dropdown provides a list of **Optimization Policies** which will be used when the model is executed. These policies are applied when lowering from a `ttir` module to an executable `ttnn` module.

### Generate C++ Code

This toggle will run the `EmitC` pass in the `tt-mlir` compiler to generate TTNN C++ Code and make it available to you after running a model. Default value for this toggle is `Off`.

## "Play" Button

![Toolbar highlighting the "execute" button](../images/tt-explorer/execute.png)

This button invokes the `execute` function which will compile and execute the model. The button will then be "loading" until execution is finished. Once execution is finished a performance trace should be overlayed on the graph and it should reload.

## "Code" Button

![Toolbar highlighting the "execute" button](../images/tt-explorer/execute.png)

If the `Generate C++ Code` option is enabled, this button will become available to view and download the C++ code in a window within explorer.

## "Logs" Button

![Toolbar highlighting the "logs" button](../images/tt-explorer/logs.png)

This button will open a window to view the shell logs while execution is running. If any errors occur they will be displayed here.

## Overridden Fields

![Example of fields with overrides enabled](../images/tt-explorer/overrides.png)

Certain Nodes on the graph will have attributes that are presented as editable fields. These are fields which have overrides available. This value can be changed and then sent to be recompiled, invalid configurations will result in errors.
