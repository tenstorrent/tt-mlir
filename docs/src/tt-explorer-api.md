# TT-Explorer

The following is a listed reference for the API in using TT-Explorer, check the TT-Adapter API reference below.

# `TTExplorer`
## Overview
The `TTExplorer` class is responsible for interacting with the model_explorer server, including uploading models, initializing settings, and executing models.

## Initialization
### **`__init__(self, port=8080, url="http://localhost", server=False, config=None)`**
Initializes the TTExplorer instance.

- Parameters:
  - `port (int)`: The port number for the model_explorer server. Default is 8080.
  - `url (str)`: The base URL for the model_explorer server. Default is `"http://localhost"`.
  - `server (bool)`: Flag to indicate if the server should be created. If this is set to true, ensure an environment where the `ttrt` and `ttmlir` python bindings is used. Default is False.
  - `config (dict)`: Configuration for the model_explorer server. Default is None.

## Methods
### `get_model_path(self, file) -> str`
Uploads a model file to the model_explorer server and returns the temporary path provided by the server.
- Parameters:
  - `file (file-like object)`: The model file to be uploaded.
- Returns:
  - `str`: The temporary path of the uploaded model file.

### **`initialize(self, settings={})`**
Initializes the server-side `TT-Explorer` by assigning a System Descriptor for future operations, **needed** to execute models.

- Parameters:
  - `settings (dict)`: Settings for initialization, currently none. Default is an empty dictionary.
- Returns:
  - `dict`: dict with `system_desc_path` key pointing to server-path to System Descriptor

### **`execute_model(self, model_path: str, settings={})`**
Executes a model on the model_explorer server with the provided settings.

- Parameters:
  - `model_path (str)`: Server path to `ttir` module to be executed, ensure that module has been uploaded first.
  - `settings (dict)`: Settings for execution. Default is an empty dictionary.
    - `"ttir_to_ttnn_options": List[str]` Pipeline options to be fed into `ttir-to-ttnn-backend-pipeline`'s String Parser
    - `"artifact_dir": str(Path)` A valid Server-Path to store artifacts from execution, if this flag is set then artifacts are not automatically deleted after execution is complete.
- Returns:
  - `dict`: Relevant emitted files from Execution
    - `"log_file": str`: Log-File from `ttrt perf` call
    - `"stdout": str`: STDOUT from `ttrt perf` call, utf-8 decoded.
    - `"perf_trace": str`: CSV Performance Trace from module run.

## Example Usage

```py
# Initialize TTExplorer
explorer = TTExplorer(server=True)
# Explorer instance now running on thread on http://localhost:8080
# Make sure you wait until the thread has started the Flask server, you can check by viewing STDOUT.

# Upload a model file
file = open('my-module.mlir', 'r')
model_path = explorer.get_model_path(file)
# Since local==server, the model_path is moved to a tempfile on the same machine

# Initialize the SystemDesc on Machine for execution purposes
explorer.initialize()

# Execute the model, store artifacts permanently in home directory.
resp = explorer.execute_model(model_path, settings={'artifact_dir': '/home/<user>/ttrt-artifacts'})

csv = resp['perf_trace'] # Do with the CSV trace as you would like to view the performance results!
```

# TT-Adapter
The following is a reference for the "REST" API provided by TT-Adapter. First, a short info-dump on how an extensible API can be built on top of Model Explorer.

## Building an API using Model Explorer
The `/apipost/v1/send_command` endpoint provides an extensible platform with which commands are sent to be executed directly by the adapter specified.  This becomes the main endpoint through which communication is facilitated between the server and client, the commands respond with an "adapter response".

### Sending Commands
The body of the command must be JSON, and only the following fields are fed into the adapter functions:
```js
cmd = {
  "extensionId": "tt_adapter", // tt_adapter to invoke functions from TT-Adapter
  "cmdId": "<name of function>", // Name of function to be run, "convert" is built into all adapters to convert some model to graph
  "modelPath": "<local_path to file>", // Path to model on server to be fed into function
  "deleteAfterConversion": False, // True if file at modelPath is to be deleted after function run
  "settings": {...}, // Object holding custom settings to be fed into function
}
```

More often than not, functions do not need all of these fields, but they must all be present to properly process the command sent into the function. Speaking of function, the function signature that all commands have to follow is as such:

```py
class TTAdapter(Adapter):
  ...
  def my_adapter_fn(self, model_path: str, settings: dict):
    pass # Parse model_path and settings objects as they are fed from send_command endpoint.
```

This function is invoked and called from a new instance every time. This is important to understand for the idea of persisting information on the server. The onus is often on the end-user to store and preserve important information such as the path of a model they've uploaded, or the paths of important artifacts that the server has produced. `TTExplorer` aims to make this as easy as possible.

Information can be processed in this function however the user would like to define, and often settings becomes a versatile endpoint to provide more information and context for the execution of some function. As an example, refer to `TTAdapter:initialize`, this function to load a SystemDesc into the environment has little to do with `modelPath` or `deleteAfterConversion`, as such these variables are not processed at all, and the function only executes a static initialization process regardless of the parameters passed into the command.

### Adapter Response
Model Explorer was probably not made to allow for such an extensible framework to be tacked onto it. As such, the adapter response is processed in a very particular way before it is sent back to the user. In particular, refer to [`model_explorer.utils.convert_adapter_response`](https://github.com/google-ai-edge/model-explorer/blob/main/src/server/package/src/model_explorer/utils.py#L40) which is run on the output of every function. This means that responses can **only** be in JSON format and are constructed as:

```js
{
  "graphs": [
    {/* response goes here */},
  ]
}
```

for custom adapter responses. This limits the transfer of raw bytes data through different MIME Types, and requires the `tt_adapter.utils.to_adapter_format` which turns any `dict` object into a model explorer adapter compatible response. While this framework works well for graphs, it makes an "extensible" API difficult to implement.

## Current API Reference:

### Initialize
Called from `TTExplorer.initialize`, used to Load SystemDesc into environment.
```js
cmd = {
  "extensionId": "tt_adapter",
  "cmdId": "initialize",
  "modelPath": "", // Irrelevant
  "deleteAfterConversion": False,
  "settings": {}, // None at the moment
}

// RESPONSE

{"system_desc_path": "<path to system_desc.ttsys>"}
```

### Execute
Called from `TTExplorer.execute_model`, executes a model.
```js
cmd = {
  "extensionId": "tt_adapter",
  "cmdId": "execute",
  "modelPath": "<server-path-to-model>",
  "deleteAfterConversion": False, // Can be set to True if TTIR module is to be deleted after run
  "settings": {
    "ttir_to_ttnn_options": List[str], // Pipeline Options to feed into ttir_to_ttnn_backend_pipeline
    "artifact_dir": str, // Path on server to store TTRT Artifacts to, artifacts are not deleted after perf if set.
  },
}

// RESPONSE

{
  "stdout": "<raw text output to STDOUT from TTRT Perf Run>",
  "log_file": "<raw logging output>",
  "perf_trace": "<raw CSV perf trace collected from TTRT Perf Run, Not present if TTRT Perf failed>",
}
```

### Convert
Standard built-in conversion function, converts TTIR Module into Model Explorer Graph. Also provides `settings` as a platform for overrides to be applied to the graph.

```js
cmd = {
  "extensionId": "tt_adapter",
  "cmdId": "convert",
  "modelPath": "<server-path-to-ttir-module>",
  "deleteAfterConversion": True/False,
  "settings": {/* Overrides */}, // Undefined at the moment
}

// RESPONSE

<model-explorer-graph JSON Object>
```
