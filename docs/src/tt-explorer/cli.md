# `tt-explorer` CLI

This page details the command-line interface (CLI) for TT-Explorer. Through the CLI, you can:
* Start the server
* Configure the server's network settings
* Control whether the UI automatically opens in a browser
* Enable or disable model execution

You can also find a summary of supported input models, so you know what can be executed and visualized through the UI.

## Input Models

Currently `tt-explorer` supports 3 types of models that can be executed/visualized.

| Input Type                             | Execution Support | Visualization Support |
| -------------------------------------- | ----------------- | --------------------- |
| `.ttnn` Flatbuffers with Debug Info    | ✔️                 | ✔️                     |
| `.ttnn` Flatbuffers without Debug Info | ❌                | ❌                    |
| `.mlir` TTIR Modules                   | ✔️                 | ✔️                     |
| `.mlir` TTNN Modules                   | ❌                | ✔️                     |

## CLI

The CLI for `tt-explorer` provides a simple suite of options to start the UI:

```bash
tt-explorer -p <port> -u <url> -q
```

## Options

<dl>
  <dt><code>-p, --port</code><dt>
  <dd>Port that model-explorer server will be exposed to. Default is 8080.</dd>

<dt><code>-u, --url</code></dt>
  <dd>Host URL Address for server. Default is "localhost".</dd>

<dt><code>-q, --no-browser</code></dt>
  <dd>Create server without opening a browser tab.</dd>

<dt><code>-x, --no-model-execution</code></dt>
  <dd>Disable execution of models from the UI.</dd>
</dl>

Example usage:

```bash
tt-explorer -p 8000 -u 0.0.0.0 -q
```

This command will start the `tt-explorer` server on port 8000, accessible at the address 0.0.0.0, and without opening a browser tab.
