# tt-adapter
Model Explorer Adapter built for TT-MLIR Compiled outputs.

## Installation
This repository depends on [tenstorrent/tt-mlir](https://github.com/tenstorrent/tt-mlir), build and activate the `venv` as the python bindings are a dependency of TT-Adapter. Run `pip install .` in the root of this repository to install `tt-adapter` into the `ttmlir_venv` environment.

## Integration into model-explorer
Model-Explorer currently primarily supports loading extensions through the CLI. An example of a run call:

```sh
model-explorer --extensions=tt_adapter
```

You should be able to see

```sh
Loading extensions...
 - ...
 - Tenstorrent Adapter
 - JSON adapter
```

in the command output to verify that it has been run.

## Testing
To test the performance trace (with dummy information) and visualization features, activate the ttmlir `venv` and run:

```sh
model-explorer test/floor_ceil_div.ttir --node_data_paths=node_data/perf_trace_dummy.json --extensions=tt_adapter
```

It will start a flask server and provide the link to visualize the model. For more information on the Model Explorer UI, refer to the [Wiki](https://github.com/google-ai-edge/model-explorer/wiki/2.-User-Guide)
