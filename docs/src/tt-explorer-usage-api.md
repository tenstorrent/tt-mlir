# `tt-explorer`
This section provides a details about the usage of `tt-explorer`.

## Input Models
Currently `tt-explorer` supports 3 types of models that can be executed/visualized.

| Input Type | Execution Support | Visualization Support |
| --- | --- | --- |
| `.ttnn` Flatbuffers with Debug Info | ✔️ | ✔️ |
| `.ttnn` Flatbuffers without Debug Info | ❌ | ❌ |
| `.mlir` TTIR Modules | ✔️ | ✔️ |
| `.mlir` TTNN Modules | ❌ | ✔️ |

## CLI
The CLI for `tt-explorer` provides a simple suite of options to start the UI:

```bash
tt-explorer -p <port> -u <url> -q
```

### Options:
- `-p, --port`: Port that model-explorer server will be exposed to. Default is 8080.
- `-u, --url`: Host URL Address for server. Default is "localhost".
- `-q, --no-browser`: Create server without opening a browser tab.

Example usage:

```bash
tt-explorer -p 8000 -u 0.0.0.0 -q
```

This command will start the `tt-explorer` server on port 8000, accessible at the address 0.0.0.0, and without opening a browser tab.

## UI
For general reference of the UI, refer to the [model-explorer wiki](https://github.com/google-ai-edge/model-explorer/wiki). This section will highlight specific UI elements added to the Tenstorrent fork of model-explorer.

### Model Execution
In the top right of the screen an additional element has been added to the top bar. It features the UI elements that invoke the execution functionality. Once the model has executed, _overlays_ are also created. These overlays provide information on how the execution went.

#### Performance Overlay
The performance overlay is generated on **every** execution, it highlights the time it took to execute each node on the graph. This is visualized with a gradient from Yellow -> Red, with Yellow being the lowest time amongst all nodes on the graph, and Red being highest.

#### Accuracy Overlay
The accuracy overlay is _only_ generated when executing from a compatible flatbuffer (`.ttnn` file extension with Debug Info). The overlay consists of either Green or Red node overlays. Green if the node passed a "golden" test, Red if not. The value for the overlay is the actual Pearson Correlation Coefficient (PCC) value with the "golden" tensor subtracted by the expected PCC value. If the number is `< 0` we know it doesn't match the expected PCC, otherwise it is an accurate comparison.

#### Advanced Settings
This menu will open a window with some advanced settings for Model execution.

##### Opt. Policy
This dropdown provides a list of **Optimization Policies** which will be used when the model is executed. These policies are applied when lowering from a `ttir` module to an executable `ttnn` module.

##### Generate C++ Code
This toggle will run the `EmitC` pass in the `tt-mlir` compiler to generate TTNN C++ Code and make it available to you after running a model. Default value for this toggle is `Off`.

#### "Play" Button
This button invokes the `execute` function which will compile and execute the model. The button will then be "loading" until execution is finished. Once execution is finished a performance trace should be overlayed on the graph and it should reload.

#### "Code" Button
If the `Generate C++ Code` flag is set, this button will become available to view and download the C++ code in a window within explorer.

#### "Comment" Button
This button will open a window to view the shell logs while execution is running. If any errors occur they will be displayed here.

### Overridden Fields
Certain Nodes on the graph will have attributes that are presented as a dropdown. These are fields which have overrides available. This value can be changed and then sent to be recompiled, invalid configurations will result in errors.

# TT-Adapter
The following is a reference for the REST API provided by TT-Adapter. First, a short info-dump on how an extensible API can be built on top of Model Explorer.

## Building an API using Model Explorer
The `/apipost/v1/send_command` endpoint provides an extensible platform with which commands are sent to be executed directly by the adapter specified.  This becomes the main endpoint through which communication is facilitated between the server and client, the commands respond with an "adapter response".

### Sending Commands
The body of the command must be JSON, and  conform to the following interface (described below as a [Typescript interface](https://www.typescriptlang.org/docs/handbook/2/everyday-types.html#interfaces)). Specific commands may narrow the field types or extend this interface.

```typescript
interface ExtensionCommand {
  cmdId: string;
  extensionId: string;
  modelPath: string;
  settings: Record<string, any>;
  deleteAfterConversion: boolean;
}
```

More often than not, functions do not need all of these fields, but they must all be present to properly process the command sent into the handling function on the server.

Speaking of function, the signature that all function that handle commands on the server have to follow is as such:

```python
class TTAdapter(Adapter):
  # ...
  def my_adapter_fn(self, model_path: str, settings: dict):
    # Parse model_path and settings objects as they are fed from send_command endpoint.
    pass
```

This function is invoked and called from a new instance every time. This is important to understand for the idea of persisting information on the server. As all requests to the server are _stateless_, the onus is often on the end-user to store and preserve important information such as the path of a model they've uploaded, or the paths of important artifacts that the server has produced. `tt-explorer` aims to make this as easy as possible.

Information can be processed in this function however the user would like to define, and often settings becomes a versatile endpoint to provide more information and context for the execution of some function. As an example, refer to `TTAdapter:initialize`, this function to load a SystemDesc into the environment has little to do with `modelPath` or `deleteAfterConversion`, as such these variables are not processed at all, and the function only executes a static initialization process regardless of the parameters passed into the command.

#### Example request

Below is an example of the JSON request sent from the UI to the server:

```json
{
  // tt_adapter to invoke functions from TT-Adapter
  "extensionId": "tt_adapter",
  // Name of function to be run, "convert" is built into all adapters to convert some model to graph
  "cmdId": "convert",
  // Path to model on server to be fed into function
  "modelPath": "/tmp/tmp80eg73we/mnist_sharding.mlir",
  // Object holding custom settings to be fed into function
  "settings": {
    "const_element_count_limit": 16,
    "edge_label_font_size": 7.5,
    "artificial_layer_node_count_threshold": 1000,
    "keep_layers_with_a_single_child": false,
    "show_welcome_card": false,
    "disallow_vertical_edge_labels": false,
    "show_op_node_out_of_layer_edges_without_selecting": false,
    "highlight_layer_node_inputs_outputs": false,
    "hide_empty_node_data_entries": false
  },
  // `true` if file at `modelPath` is to be deleted after function run
  "deleteAfterConversion": true
}
```

### Adapter Response
Model Explorer was probably not made to allow for such an extensible framework to be tacked onto it. As such, the adapter response is processed in a very particular way before it is sent back to the user. In particular, refer to [`model_explorer.utils.convert_adapter_response`](https://github.com/google-ai-edge/model-explorer/blob/main/src/server/package/src/model_explorer/utils.py#L40) which is run on the output of every function.

This means that for compatibility reasons (i.e. to not stray too much from the upstream implementation that we are based off of) responses sent from the server must be in JSON format **only** and wrap the data on a `graph` property.

Below is the base typescript interface that the UI expects for the json response. Commands can define custom data _inside_ the `graph` property.

```typescript
/** A response received from the extension. */
interface ExtensionResponse<
  G extends Array<unknown> = Graph[],
  E extends unknown = string
> {
  graphs: G;
  error?: E;
}
```

For custom adapter responses. This limits the transfer of raw bytes data through different MIME Types, and requires the `tt_adapter.utils.to_adapter_format` which turns any `dict` object into a model explorer adapter compatible response. While this framework works well for graphs, it makes an "extensible" API difficult to implement.

## Current API Reference:

### Convert
Standard built-in conversion function, converts TTIR Module into Model Explorer Graph. Also provides `settings` as a platform for overrides to be applied to the graph.
#### Request

```typescript
// As this is the base request everything is based off,
// this interface only narrows down the command to be "convert".
interface AdapterConvertCommand extends ExtensionCommand {
  cmdId: 'convert';
}
```

#### Response
```typescript
// As this is the base response everything is based off,
// it is exactly the same as `ExtensionResponse`.
type AdapterConvertResponse = ExtensionResponse;
```

```json
{
  "graphs": [{
    // Model Explorer Graph JSON Object
  }]
}
```

### Initialize
Called from `TTExplorer.initialize`, used to Load SystemDesc into environment.

#### Request

```typescript
interface InitializeCommand extends ExtensionCommand {
  cmdId: 'initialize';
}
```

#### Response

```typescript
type AdapterInitializeResponse = ExtensionResponse<[{
  system_desc_path: string
}]>;
```

```json
{
  "graphs": [{
    "system_desc_path": "<path to system_desc.ttsys>"
  }]
}
```

### Execute
Called from `TTExplorer.execute_model`, executes a model.

#### Request

```typescript
interface AdapterExecuteCommand extends ExtensionCommand {
  cmdId: 'execute';
}
```

#### Response
```typescript
// When the request is successful, we don't expect any response back.
// Thus, an empty array is returned for `graphs`.
type AdapterExecuteResponse = ExtensionResponse<[]>;
```

```json
{
  "graphs": []
}
```

### Status Check

Called from `...`, it is used for checking the execution status of a model and update the UI accordingly.

#### Request

```typescript
interface AdapterStatusCheckCommand extends ExtensionCommand {
  cmdId: 'status_check';
}
```

#### Response
```typescript
type AdapterStatusCheckResponse = ExtensionResponse<[{
  isDone: boolean;
  progress: number;
  total?: number;
  timeElapsed?: number;
  currentStatus?: string;
  error?: string;
  stdout?: string;
  log_file?: string;
}]>;
```

```json
{
  "graphs": [{
    "isDone": false,
    "progress": 20,
    "total": 100,
    "timeElapsed": 234,
    "stdout": "Executing model...\nPath: /path/to/model",
    "log_file": "/path/to/log/on/the/server"
  }]
}
```
### Override

Called from `...` to send overrides made through the UI to the server for processing.

#### Request

```typescript
interface KeyValue {
  key: string;
  value: string;
}

interface AdapterOverrideCommand extends ExtensionCommand {
  cmdId: 'override';
  settings: {
    graphs: Graph[];
    overrides: Record<string, {
      named_location: string,
      attributes: KeyValue[]
    }>;
  };
}
```

#### Response
```typescript
type AdapterOverrideResponse = ExtensionResponse<[{
  success: boolean;
}]>;
```

```json
{
  "graphs": [{
    "success": true
  }]
}
```

## Editable attributes

To enable an attribute to be edited, a response coming from the server should contain the `editable` field on the attribute.

The typescript interface is as follows:
```typescript
interface Graph {
	nodes: GraphNode[];
	// ...
}

interface GraphNode {
	attrs?: Attribute[];
	// ...
}

type EditableAttributeTypes = EditableIntAttribute | EditableValueListAttribute | EditableGridAttribute; // Attribute types are defined below...

interface Attribute {
	key: string;
	value: string;
	editable?: EditableAttributeTypes; // <- the editable attribute information
}
```

### `EditableIntAttribute`

This editable attribute represents a list of integer values. It expects the **attribute `value`** to be formatted as a string, starting with `[` and ending with `]`, with all values separated by `,`. Like the example below:
```
[1, 2, 3]
```

The typescript interface for the `editable` attribute is this:
```typescript
interface EditableIntAttribute {
	input_type: 'int_list';
	min_value?: number = 0;
	max_value?: number = 100;
	step?: number = 1;
}
```

Both `min_value` and `max_value` define the accepted range of values, and `step` define the number to increment or decrement per step.

The default range of values is between `0` and `100`, inclusive, and the default step is `1`. Thus by default, the value will increment or decrement by `1` each time to a minimum of `0` and a maximum of `100`.

Here is an example of what this attribute look like:
```json
{
  "graphs": [{
    "nodes": [
	    {
		    "attrs": [
			    {
				    "key": "shape",
				    "value": "[8, 8]",
				    "editable": {
					    "input_type": "int_list",
					    "min_value": 8,
					    "max_value": 64,
					    "step": 8
				    }
			    }
		    ]
	    }
    ]
  }]
}
```

### `EditableValueListAttribute`

This editable attribute define a fixed list of string values to display.

The typescript interface for the `editable` attribute is this:
```typescript
interface EditableValueListAttribute {
	input_type: 'value_list';
	options: string[];
}
```

The `options` property provides the list of options to be displayed.  The current value will be added to this list and any duplicates will be removed.

Here is an example of what this attribute look like:
```json
{
  "graphs": [{
    "nodes": [
	    {
		    "attrs": [
			    {
				    "key": "chip_arch",
				    "value": "wormhole",
				    "editable": {
					    "input_type": "value_list",
					    "options": [
						    "wormhole",
						    "grayskull"
					    ]
				    }
			    }
		    ]
	    }
    ]
  }]
}
```

### `EditableGridAttribute`

The grid attribute is similar to to the integer list, with the main difference that you can specify a `separator` for the place the list will be split, and it doesn't need to be enclosed in bracket (`[` and `]`). The data for a grid attribute looks like this:
```
4x4x2
```

The typescript interface for the `editable` attribute is this:
```typescript
interface EditableGridAttribute {
	input_type: 'grid';
	separator?: string = 'x';
	min_value?: number = 0;
	max_value?: number = 100;
	step?: number = 1;
}
```

Both `min_value` and `max_value` define the accepted range of values, and `step` define the number to increment or decrement per step.

The default range of values is between `0` and `100`, inclusive, and the default step is `1`. Thus by default, the value will increment or decrement by `1` each time to a minimum of `0` and a maximum of `100`.

The `separator` attribute defines the character used to split the string, it defaults to "`x`".

Here is an example of what this attribute look like:
```json
{
  "graphs": [{
    "nodes": [
	    {
		    "attrs": [
			    {
				    "key": "grid",
				    "value": "4x4",
				    "editable": {
					    "input_type": "grid",
					    "min_value": 4,
					    "max_value": 64,
					    "step": 4,
					    "separator": "x"
				    }
			    }
		    ]
	    }
    ]
  }]
}
```
