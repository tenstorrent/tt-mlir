# tt-adapter
Model Explorer Adapter built for TT-MLIR outputs. Contains the logic for converting IRs into model explorer graphs.

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
