# TT-Explorer - API

## TT-Adapter

The following is a reference for the REST API provided by TT-Adapter.

First, a short info-dump on how an extensible API can be built on top of Model Explorer.

### Building an API using Model Explorer

The `/apipost/v1/send_command` endpoint provides an extensible platform with which commands are sent to be executed directly by the adapter specified. This becomes the main endpoint through which communication is facilitated between the server and client, the commands respond with an "adapter response".

#### Sending Commands

The body of the command must be JSON, and conform to the following interface (described below as a [Typescript interface](https://www.typescriptlang.org/docs/handbook/2/everyday-types.html#interfaces)). Specific commands may narrow the field types or extend this interface providing extra information. But all interfaces should be based on this.

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

On the server side, the signature that all function that handle commands have to follow is:

```python
class TTAdapter(Adapter):
  # ...
  def my_adapter_fn(self, model_path: str, settings: dict):
    # Parse model_path and settings objects as they are fed from send_command endpoint.
    pass
```

This function is invoked and called from a new instance every time. This is important to understand for the idea of persisting information on the server. As all requests to the server are _stateless_, the onus is often on the end-user to keep track of important information such as the path of a model they've uploaded, or the paths of important artifacts that the server has produced. `TTExplorer` aims to make this as easy as possible, but this may not always be possible due to the very nature of how the server works.

Information can be processed in this function as defined by the user, and often settings becomes a versatile endpoint to provide more information and context for the execution of some function. As an example, refer to `ModelRunner:initialize`, this function doesn't use any of the parameter, as such they are not processed at all, and the function only executes a static initialization process regardless of the parameters passed into the command.

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

Model Explorer was not made to allow for such an extensible framework to be tacked onto it. As such, the adapter response is processed in a very particular way before it is sent back to the user.

In particular, refer to [`model_explorer.utils.convert_adapter_response`](https://github.com/google-ai-edge/model-explorer/blob/main/src/server/package/src/model_explorer/utils.py#L40) which is run on the output of every function.

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

## Current API Reference

### `convert`

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

### `initialize`

Called from `TTAdapter.__init__`, used to Load SystemDesc into environment.

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

### `execute`

Called from `TTAdapter.execute`, executes a model.

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

### `status-check`

Called from `TTExplorer.status_check`, it is used for checking the execution status of a model and update the UI accordingly.

#### Request

```typescript
interface AdapterStatusCheckCommand extends ExtensionCommand {
	cmdId: 'status_check';
}
```

#### Response

```typescript
type AdapterStatusCheckResponse = ExtensionResponse<[{
	isDone: boolean,
	progress: number,
	total?: number,
	timeElapsed?: number,
	currentStatus?: string,
	error?: string,
	stdout?: string,
	log_file?: string
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
	// ...
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

The `options` property provides the list of options to be displayed. The current value will be added to this list and any duplicates will be removed.

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

## Attribute display type

To change how the attribute is displayed from plain text to something else, we do extend the attribute interface (presented above) with the `display_type` optional field.

```typescript
type AttributeDisplayType = 'memory';

interface Attribute {
	key: string;
	value: string;
	display_type?: AttributeDisplayType; // <- Optional, add a different display type.
	// ...
}
```

If the `display_type` attribute is present, and it matches one of the available values, then the attribute will display differently than the others.

In the example below, the two attributes have different display types, one shows the regular, plain text display; and the other shows the `memory` display type, which renders it as a progress bar.

![Example of different attribute display types](../images/tt-explorer/display-types.png)

### `memory`

Setting the display type to `memory` will make the attribute try to render as a progress bar.

The UI will then check the `value` property in the attribute for the following conditions:

- Is a [double precision floating point number](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number#number_encoding)
- Is not [`NaN`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/NaN#observably_distinct_nan_values)
- Is grater than or equal to `0`
- Is less than or equal to `1`

If all of the conditions are true, then the `value` will be rendered as a progress bar.
