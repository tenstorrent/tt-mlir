# Step 1: Check if TTMLIR_TOOLCHAIN_DIR is already set, if not, set it to /opt/ttmlir-toolchain
if test -z "$TTMLIR_TOOLCHAIN_DIR"
  set -gx TTMLIR_TOOLCHAIN_DIR "/opt/ttmlir-toolchain"
end

# Step 2: Check if TTMLIR_VENV_DIR is already set, if not, set it to the venv directory inside TTMLIR_TOOLCHAIN_DIR
if test -z "$TTMLIR_VENV_DIR"
  set -gx TTMLIR_VENV_DIR "$TTMLIR_TOOLCHAIN_DIR/venv"
end

if test -f $TTMLIR_VENV_DIR/bin/activate.fish
  source $TTMLIR_VENV_DIR/bin/activate.fish
end

set -gx TTMLIR_ENV_ACTIVATED 1
set -gx PATH (pwd)/build/bin $TTMLIR_TOOLCHAIN_DIR/bin $TTMLIR_TOOLCHAIN_DIR/venv/bin $PATH
set -gx TT_METAL_RUNTIME_ROOT (pwd)/third_party/tt-metal/src/tt-metal
set -gx TT_METAL_BUILD_HOME (pwd)/third_party/tt-metal/src/tt-metal/build
set -gx TT_MLIR_HOME (pwd)
set -gx PYTHONPATH (pwd)/build/python_packages:(pwd)/build/runtime/python:(pwd)/.local/toolchain/python_packages/mlir_core
set -gx PYTHONPATH $PYTHONPATH:$TT_METAL_RUNTIME_ROOT:$TT_METAL_RUNTIME_ROOT/ttnn:$TT_METAL_RUNTIME_ROOT/tt_eager:$TT_METAL_BUILD_HOME/tools/profiler/bin  # Metal paths
