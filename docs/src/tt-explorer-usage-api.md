# TT-Explorer
This section provides a details about the usage of TT-Explorer.

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

This command will start the TT-Explorer server on port 8000, accessible at the address 0.0.0.0, and without opening a browser tab.

## UI
For general reference of the UI, refer to the [model-explorer wiki](https://github.com/google-ai-edge/model-explorer/wiki). This section will highlight specific UI elements added to the tenstorrent fork of model-explorer.

### Model Execution
In the top right of the screen an additional element has been added to the top bar. It features the UI elements that invoke the execution functionality.

#### Opt. Policy
This dropdown provides a list of **Optimization Policies** which will be used when the model is executed. These policies are applied when lowering from a `ttir` module to an executable `ttnn` module.

#### "Upload" Button
Once Overriden Fields have been changed or modified, this button will be available to send the overrides to the backend. The overrides will then be processed and the module recompiled to include these new changes.

#### "Play" Button
This button invokes the `execute` function which will compile and execute the model. The button will then be "loading" until execution is finished. Once execution is finished a performance trace should be overlayed on the graph and it should reload.

#### "Comment" Button
This button will open a window to view the shell logs while exeuction is running. If any errors occur they will be displayed here.

### Overriden Fields
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

This function is invoked and called from a new instance every time. This is important to understand for the idea of persisting information on the server. As all requests to the server are _stateless_, the onus is often on the end-user to store and preserve important information such as the path of a model they've uploaded, or the paths of important artifacts that the server has produced. `TTExplorer` aims to make this as easy as possible.

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
