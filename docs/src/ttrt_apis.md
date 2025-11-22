# Using ttrt programmatically
This document explains how to use ttrt as a Python package via the pre-built wheel generated during the tt-mlir build (when you have runtime or perf flags enabled). Using the wheel provides programmatic access to ttnn compilation, execution, and result inspection from Python scripts-- without needing to invoke the command line interface (CLI) for every operation. 

With the wheel, you can:
* Run ttnn binaries directly from Python
* Configure logging and artifact directories programmatically
* Initialize and inspect tensors, including golden checks and memory reports
* Register Python callback hooks to extend runtime behavior for debugging, tracing, or custom instrumentation

>**NOTE:** If you prefer to work exclusively from the command line, see the [ttrt CLI documentation](ttrt.md). This page focuses entirely on using the Python wheel and ttrt APIs. 

## Prerequisites

Ensure you meet the prerequisites in the [Prerequisites section](ttrt.md#prerequisites) of the ttrt documentation.

## Installing ttrt as a Python wheel
Every time you build `ttrt` as part of the tt-mlir build, a custom Python wheel is generated in `build/tools/ttrt/build`. 

The wheel packages the compiled ttnn runtime along with the Python API bindings, so you can install and use it in any Python environment or Docker container without needing the full tt-mlir build. 

>**NOTE:** The wheel reflects the configuration and build flags used for tt-mlir, so it is tied to your local build.

To install the wheel, do the following: 

1. Follow the [build instructions](ttrt.md#building) in the ttrt documentation.

2. Download the wheel from `build/tools/ttrt/build`. The file name should look something like: `ttrt-0.0.235-cp311-cp311-linux_x86_64.whl`. 

3. Create a python venv:
```bash
python -m venv ttrt_env
source ttrt_env/bin/activate
```

4. Install the wheel (replace the example command with your version of the wheel):
```bash
pip install build/tools/ttrt/build/ttrt-0.0.235-cp311-cp311-linux_x86_64.whl
```

>**NOTE:** After installing the wheel, you can use `ttrt` programmatically in Python without having the full tt-mlir build present. 

You can now import `ttrt` as a package for your scripts: 

```python
from ttrt.common.api import API
```

## Working with the ttrt Python API 
Using the ttrt Python wheel follows a simple workflow: 

1. Import the ttrt APIs and utilities
2. Initialize all the API registrations 
3. Provide arguments
4. (Optional) Configure logging
5. (Optional) Configure artifacts location 
6. Create an API instance 
7. (Optional) Register runtime callback hooks 
8. Execute the API and inspect results 

Here is a sample template showing the generic workflow: 

```python
# 1. Import APIs and utilities
# These are needed for creating API instances, logging, artifacts, and optional runtime hooks
from ttrt.common.api import API
from ttrt.common.util import Logger, Artifacts
import ttrt.runtime  # only needed if using callbacks

# 2. Initialize APIs
# Registers all available API classes so they can be used in your script
API.initialize_apis()

# 3. Provide arguments
# Arguments mirror CLI flags; any missing args will use defaults
custom_args = {}
custom_args["--clean-artifacts"] = True
custom_args["--save-artifacts"] = True
custom_args["--loops"] = 10
custom_args["--init"] = "randn"
custom_args["binary"] = "/path/to/subtract.ttnn"

# 4. Configure logging (optional)
# Allows API instances to log to specific files at a given verbosity level
log_file_name = "example.log"
custom_logger = Logger(log_file_name)

# 5. Configure artifacts (optional)
# Sets where execution metadata and generated files are stored
artifacts_folder_path = "/opt/folder"
custom_artifacts = Artifacts(logger=custom_logger, artifacts_folder_path=artifacts_folder_path)

# 6. Create API instance
# Each API class corresponds to a CLI subcommand (Query, Read, Run)
run_instance = API.Run(args=custom_args, logger=custom_logger, artifacts=custom_artifacts)

# 7. Register callbacks (optional)
# Hooks allow you to run custom Python functions before/after each MLIR op
callback_env = ttrt.runtime.DebugHooks.get(pre_op_callback, post_op_callback)

# 8. Execute
# Returns a result code and structured results
result_code, results = run_instance()
```

### Python API Reference
This section details what ttrt APIs are available and how to use them. Some APIs are fully available in the ttrt CLI as well as programmatically, while some are partially or entirely available only programmatically. Here is a chart summarizing availability: 

| Category | Python API | Notes |
|----------|------------|-------|
| Essential | Query | Mirrors `ttrt query` |
|  | Read | Mirrors `ttrt read` |
|  | Run | Mirrors `ttrt run`; Python API adds optional logging, artifacts, runtime hooks |
|  | Perf | Mirrors `ttrt perf` |
|  | Check | Mirrors `ttrt check` |
|  | EmitPy | Mirrors `ttrt emitpy`; Python API adds optional logging, artifacts, runtime hooks |
|  | EmitC | Mirrors `ttrt emitc`; Python API adds optional logging, artifacts, runtime hooks |
| Multi-Device Deployment | Device Discovery | `getNumAvailableDevices()` – count of devices |
|  | Mesh Management | `openMeshDevice()`, `closeMeshDevice()`, `reshapeMeshDevice()` – create/close/reshape device meshes |
|  | Sub-Mesh Operations | `createSubMeshDevice()`, `releaseSubMeshDevice()` – data-parallel sub-mesh handling |
| Tensor Pool Manipulation | Tensor Creation & Management | Host/multi-device tensor creation, `toHost()`, `toLayout()`, `memcpy()`, `deallocateTensor()` |
|  | Tensor Inspection | `getTensorShape()`, `getTensorStride()`, `getTensorDataType()`, `getTensorVolume()` |
| Callback Registration | Operation Hooks | `DebugHooks.get(preOp, postOp)` – register callbacks for runtime inspection; access op metadata and tensor outputs |
| Environment Config | Runtime & Device Selection | `setMlirHome()`, `setMetalHome()`, `get/setCurrentDeviceRuntime()`, `get/setCurrentHostRuntime()` |
| Distributed Runtime | Multi-node Execution | `launchDistributedRuntime()`, `shutdownDistributedRuntime()` |
| Memory Profiling | Memory Inspection | `getMemoryView()`, `dumpMemoryReport()` – check memory usage per buffer type |
| Performance Tracing | Device Profiling | `readDeviceProfilerResults()` – device profiling with optional perf trace |
| Synchronization | Event/Tensor Wait | `wait(event)`, `wait(tensor)` – synchronize execution |
| Home Directory Setters |  | Can configure artifact/log directories via `Artifacts` and `Logger` utilities |

For more details about the programmatic-only APIs in the table, you can review the code here:
* [runtime.h](https://github.com/tenstorrent/tt-mlir/blob/0abe29ab/runtime/include/tt/runtime/runtime.h)
* [runtime.cpp](https://github.com/tenstorrent/tt-mlir/blob/0abe29ab/runtime/lib/runtime.cpp)


#### CLI-Mirroring APIs 
This section goes over how to reference details for CLI-mirroring APIs, usage patterns with Python, and explains the additional features available through the Python APIs. 

For details about what commands are available for the CLI-mirroring APIs, refer to the [ttrt command line commands](ttrt.md#ttrt-command-line-commands) section of the ttrt documentation. 

Here is a sample displaying how to use CLI-mirrored features: 

```python
from ttrt.common.api import API

# Initialize all available APIs
API.initialize_apis()

# Arguments mirror CLI flags
args = {
    "--verbose": True,  # example CLI flag
    "--device": "0"     # specify device index, matches CLI
}

# Create a Query API instance
query_instance = API.Query(args=args)

# Execute the API
result_code, results = query_instance()

# Inspect results
print("Result code:", result_code)
print("Query results:", results)
```

##### Python-specific features of CLI-mirroring APIs 
In addition to the CLI-mirrored commands, the `run`, `emitpy`, and `emitc` APIs allow you to do custom logging, create custom artifacts directories, and runtime callback hooks. 

**Custom Logging**
You can specify a specific logging module you want to set inside your API instance. The rationale behind this is to support different instances of different APIs, all being able to be logged to a different file. You can also customize the level of detail your log file contains.

```python
from ttrt.common.util import Logger
import os

os.environ["LOGGER_LEVEL"] = "DEBUG"
log_file_name = "some_file_name.log"
custom_logger = Logger(log_file_name)
read_instance = API.Read(logger=custom_logger)
```

**Custom Artifacts**
You can specify a specific artifacts directory to store all the generate metadata during the execution of any API run. This allows you to specify different artifact directories if you wish for different instances of APIs.

```python
from ttrt.common.util import Artifacts

log_file_name = "some_file_name.log"
artifacts_folder_path = "/opt/folder"
custom_logger = Logger(log_file_name)
custom_artifacts = Artifacts(logger=custom_logger, artifacts_folder_path=artifacts_folder_path)
run_instance = API.Run(artifacts=custom_artifacts)
```

**Runtime callback hooks**
For details about runtime callback hooks, please see the section [Runtime integration](#runtime-integration) below. These hooks may be used with `run`, `emitpy`, and `emitc` APIs as well as any custom Python code that imports `ttrt.runtime`. 

## Runtime integration
The full set of `ttrt.runtime` exposed APIs and types can be found in `runtime/python/runtime/runtime.cpp`, however only the ones intended to be used for runtime customization through callback hooks are outlined here.

### Callback hooks
MLIR Runtime exposes a feature to register python callback functions. Any two python fuctions can be provided - the first function will be executed before every op in MLIR Runtime, the second after every op. The following steps describe how to extend your application to register python functions. Callback functions are already implemented by default for pbd debugger implementation and gathering memory and golden check data as outlined in the `run` API section.

1. Pybind DebugHooks C++ class, specifically `tt::runtime::debug::Hooks::get`. See `runtime/python/runtime/runtime.cpp` for an example of how `ttrt` pybinds it.
```cpp
tt::runtime::debug::Hooks
tt::runtime::debug::Hooks::get
```

2. Register callback functions in your python script. The following is registering the two callback functions written in `tools/ttrt/common/callback.py`. The Debug Hooks get function has been pybinded to `ttrt.runtime.DebugHooks.get`
```python
import ttrt.runtime

callback_env = ttrt.runtime.DebugHooks.get(pre_op_callback_runtime_config, post_op_callback_runtime_config)
```

3. The callback function has a particular function signature, which looks like the following
```python
def pre_op_callback_runtime_config(binary, program_context, op_context):
```
| Parameter | Type | Description |
|-----------|------|-------------|
| `binary` | `ttrt.binary.Binary` | Reference to the binary currently running |
| `program_context` | `ttrt.runtime.ProgramContext` | Reference to the program currently executing |
| `op_context` | `ttrt.runtime.OpContext` | Reference to the current MLIR operation |

>**Notes:**
>* The callback function is executed before or after each MLIR op depending on registration.
>* It does not need to return a value.
>* Certain runtime APIs (e.g., getting op output tensors) can only be called inside the callback since they rely on `op_context`.

4. Each of these parameters has certain runtime APIs exposed which can only be called within the callback functions since they rely on the `op_context` variable that is only available from runtime during callbacks.
```python
import ttrt.runtime

loc = ttrt.runtime.get_op_loc_info(op_context) # get the location of the op as a string which is used as the key when indexing the golden tensors stored in the flatbuffer
op_debug_str = ttrt.runtime.get_op_debug_str(op_context) # get the op debug str (contains op metadata inculding op type, attributes, input tensor shapes and dtypes, memref with layout and buffer type, and loc)
op_golden_tensor = ttrt.runtime.get_debug_info_golden(binary, loc) # get the golden tensor from the binary as a ttrt.binary GoldenTensor object
op_output_tensor = ttrt.runtime.get_op_output_tensor(op_context, program_context) # get the currently running output tensor from device as a ttrt.runtime Tensor object, if this is called in a preOp function or the op doesn't output a tensor, an empty tensor will be returned.
```

>**NOTE:** `ttrt` is not needed to implement this callback feature. It aims to provide an example of how this callback feature can be implemented for golden application.

### Putting it all together
This example combines CLI-mirroring API usage with custom logging, artifacts, and callback hooks.
You can do interesting stuff when combining all the above features into your python script:

```python
from ttrt.common.api import API
from ttrt.common.util import Logger
from ttrt.common.util import Artifacts

API.initialize_apis()

custom_args = {}
custom_args["--clean-artifacts"] = True
custom_args["--save-artifacts"] = True
custom_args["--loops"] = 10
custom_args["--init"] = "randn"
custom_args["binary"] = "/path/to/subtract.ttnn"

log_file_name = "some_file_name.log"
custom_logger = Logger(log_file_name)

artifacts_folder_path = "/opt/folder"
custom_artifacts = Artifacts(logger=custom_logger, artifacts_folder_path=artifacts_folder_path)

run_instance = API.Run(args=custom_args, logger=custom_logger, artifacts=custom_artifacts)
result_code, results = run_instance()
```