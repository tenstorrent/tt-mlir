# TT-explorer

TT-explorer is a tool for MLIR graph visualization and applying optimizer overrides in order to easily experiment with model performance.

TODO: add documentation from old tt-explorer repo

## Build
```bash
source env/activate
cmake --build build -- explorer
```

## Usage
Start the server with:
```bash
tt-explorer
```

Then open http://localhost:8080 in the browser.

#### Port Forwarding
P.S.
If using a remote machine make sure to forward the 8080 port. E.g:
```bash
ssh -L 8080:localhost:8080 user@remote-machine
```
Or set the "Tt › Ird › Reservation: Ports" setting in vscode-ird.
