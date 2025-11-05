# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import model_explorer

import requests
import time
import multiprocessing
import pytest
import glob
import os
import logging
import portpicker
import sys

# Debug: Print environment information at test startup
print("\n" + "=" * 80)
print("DEBUG: Environment Information")
print("=" * 80)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"BUILD_DIR env var: {os.environ.get('BUILD_DIR', 'NOT SET')}")
print(f"INSTALL_DIR env var: {os.environ.get('INSTALL_DIR', 'NOT SET')}")
print(f"IMAGE_NAME env var: {os.environ.get('IMAGE_NAME', 'NOT SET')}")
print(f"TT_METAL_RUNTIME_ROOT: {os.environ.get('TT_METAL_RUNTIME_ROOT', 'NOT SET')}")

# Check key paths
build_dir = os.environ.get("BUILD_DIR", "build")
install_dir = os.environ.get("INSTALL_DIR", "install")
print(f"\nDEBUG: Path checks:")
print(f"  BUILD_DIR exists: {os.path.exists(build_dir)}")
print(f"  INSTALL_DIR exists: {os.path.exists(install_dir)}")

# Check for tracy-specific indicators
tracy_indicators = [
    "/usr/bin/tracy",
    "/usr/local/bin/tracy",
    os.path.join(install_dir, "lib/libtracy.so"),
]
print(f"\nDEBUG: Tracy build indicators:")
for path in tracy_indicators:
    print(f"  {path}: {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")

# Check runtime libraries
runtime_libs = [
    "ttrt",
    "tt-mlir",
    "model_explorer",
]
print(f"\nDEBUG: Python packages:")
for lib in runtime_libs:
    try:
        mod = __import__(lib.replace("-", "_"))
        print(
            f"  {lib}: {getattr(mod, '__version__', 'version unknown')} at {getattr(mod, '__file__', 'unknown path')}"
        )
    except ImportError as e:
        print(f"  {lib}: NOT FOUND ({e})")

# Check if perf tracing is enabled
perf_env_vars = [
    "TT_RUNTIME_ENABLE_PERF_TRACE",
    "TTNN_CONFIG_OVERRIDES",
    "TT_METAL_LOGGER_LEVEL",
    "TT_MLIR_HOME",
    "TT_METAL_HOME",
    "TT_METAL_RUNTIME_ROOT",
]
print(f"\nDEBUG: Performance-related env vars:")
for var in perf_env_vars:
    print(f"  {var}: {os.environ.get(var, 'NOT SET')}")

# Check for Tracy perf capture tools (needed for ttrt perf)
# TT_METAL_RUNTIME_ROOT is the new replacement for TT_METAL_HOME
tt_metal_runtime_root = os.environ.get("TT_METAL_RUNTIME_ROOT", "")
tt_metal_home = os.environ.get("TT_METAL_HOME", "")
tracy_tools = []

# Check TT_METAL_RUNTIME_ROOT first (new standard)
if tt_metal_runtime_root:
    tracy_tools.extend([
        os.path.join(tt_metal_runtime_root, "capture-release"),
        os.path.join(tt_metal_runtime_root, "csvexport-release"),
    ])

# Check TT_METAL_HOME (legacy)
if tt_metal_home:
    tracy_tools.extend([
        os.path.join(tt_metal_home, "capture-release"),
        os.path.join(tt_metal_home, "csvexport-release"),
    ])

# If neither env var is set, check common locations
if not tt_metal_runtime_root and not tt_metal_home:
    tracy_tools = [
        "/opt/tt_metal/capture-release",
        "/opt/tt_metal/csvexport-release",
        "/usr/local/bin/capture-release",
        "/usr/local/bin/csvexport-release",
    ]

print(f"\nDEBUG: Tracy perf tools (required for performance data):")
for tool_path in tracy_tools:
    exists = os.path.exists(tool_path)
    executable = os.access(tool_path, os.X_OK) if exists else False
    print(f"  {tool_path}: {'EXISTS' if exists else 'NOT FOUND'} {'(executable)' if executable else '(not executable)' if exists else ''}")

# Check for ttrt-artifacts directory (where perf data is stored)
tt_mlir_home = os.environ.get("TT_MLIR_HOME", "")
if tt_mlir_home:
    artifacts_dir = os.path.join(tt_mlir_home, "ttrt-artifacts")
    print(f"\nDEBUG: ttrt-artifacts directory:")
    print(f"  Path: {artifacts_dir}")
    print(f"  Exists: {os.path.exists(artifacts_dir)}")
    if os.path.exists(artifacts_dir):
        print(f"  Writable: {os.access(artifacts_dir, os.W_OK)}")
        # List contents if it exists
        try:
            contents = os.listdir(artifacts_dir)
            print(f"  Contents: {contents if contents else '(empty)'}")
        except Exception as e:
            print(f"  Error listing contents: {e}")

print("=" * 80 + "\n")

HOST = "localhost"
# Use portpicker to pick a port for us. (say that 10 times fast)
PORT = portpicker.pick_unused_port()
COMMAND_URL = "http://" + HOST + ":" + str(PORT) + "/apipost/v1/send_command"
TEST_LOAD_MODEL_PATHS = [
    "test/ttmlir/Explorer/**/*.mlir",
    "test/ttmlir/Silicon/TTNN/n150/perf/**/*.mlir",
]
MNIST_SHARDING_PATH = "test/ttmlir/Silicon/TTNN/n150/optimizer/mnist_sharding.mlir"
MNIST_STABLEHLO_PATH = "test/ttmlir/Silicon/StableHLO/n150/mnist_inference.mlir"
TEST_EXECUTE_MODEL_PATHS = [
    MNIST_SHARDING_PATH,
]

if "TT_EXPLORER_GENERATED_MLIR_TEST_DIRS" in os.environ:
    for path in os.environ["TT_EXPLORER_GENERATED_MLIR_TEST_DIRS"].split(","):
        if os.path.exists(path):
            TEST_LOAD_MODEL_PATHS.append(path + "/**/*.mlir")
        else:
            logging.error(
                "Path %s provided in TT_EXPLORER_GENERED_MLIR_TEST_DIRS doesn't exist. Tests not added.",
                path,
            )

if "TT_EXPLORER_GENERATED_TTNN_TEST_DIRS" in os.environ:
    for path in os.environ["TT_EXPLORER_GENERATED_TTNN_TEST_DIRS"].split(","):
        if os.path.exists(path):
            TEST_LOAD_MODEL_PATHS.append(path + "/**/*.ttnn")
        else:
            logging.error(
                "Path %s provided in TT_EXPLORER_GENERED_TTNN_TEST_DIRS doesn't exist. Tests not added.",
                path,
            )

FILTERED_TESTS = [
    # This test is way too large to fit reasonably in CI.
    "test_llama_attention.ttnn",
    "test_hoisted_add.ttnn",
]


def get_test_files(paths):
    files = []
    for path in paths:
        files.extend(glob.glob(path, recursive=True))

    files = [
        file for file in files if all(not file.endswith(x) for x in FILTERED_TESTS)
    ]

    return files


def GET_TTNN_TEST():
    for test in get_test_files(TEST_LOAD_MODEL_PATHS):
        if test.endswith("test_mnist[ttnn-28x28_digits-f32].ttnn"):
            return test
    return None


@pytest.fixture(scope="function", autouse=True)
def start_server(request):
    """Start the model explorer server before running tests and stop it after."""
    server_thread = multiprocessing.Process(
        target=model_explorer.visualize_from_config,
        kwargs={
            "extensions": ["tt_adapter"],
            "host": HOST,
            "port": PORT,
            "no_open_in_browser": True,
        },
    )
    server_thread.start()

    # Wait for the server to start
    for _ in range(200):  # Try for up to 20 seconds
        try:
            response = requests.get(f"http://{HOST}:{PORT}/check_health", timeout=1)
            if response.status_code == 200:
                print("Explorer server started")
                break
        except requests.ConnectionError:
            pass
        finally:
            time.sleep(0.1)
    else:
        raise RuntimeError("Server did not start within the expected time")

    # Terminate the server and wait for it to finish.
    def server_shutdown():
        server_thread.terminate()
        server_thread.join()

    request.addfinalizer(server_shutdown)


def send_command(command, model_path, settings={}):
    cmd = {
        "extensionId": "tt_adapter",
        "cmdId": command,
        "modelPath": model_path,
        "deleteAfterConversion": False,
        "settings": settings,
    }

    return requests.post(COMMAND_URL, json=cmd, timeout=10)


def execute_command(model_path, settings):
    result = send_command("execute", model_path, settings)
    assert result.ok
    if "error" in result.json():
        print(result.json())
        assert False


def wait_for_execution_to_finish(timeout):
    for _ in range(timeout):
        try:
            response = send_command("status_check", "")
            if response.status_code == 200 and response.json().get("graphs")[0].get(
                "isDone"
            ):
                return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            raise Exception("Status check request failed")
        time.sleep(1)
    raise RuntimeError(f"Execution did not finish within {timeout} seconds")


def execute_command_and_wait(model_path, settings, timeout):
    execute_command(model_path, settings)
    adapter_response = wait_for_execution_to_finish(timeout)
    assert "graphs" in adapter_response
    assert len(adapter_response["graphs"]) == 1
    response = adapter_response["graphs"][0]
    assert response["isDone"]
    assert response["error"] is None


def convert_command_and_assert(model_path):
    result = send_command("convert", model_path)
    assert result.ok
    if "error" in result.json():
        print(result.json())
        assert False
    return result.json()


@pytest.mark.parametrize("model_path", get_test_files(TEST_LOAD_MODEL_PATHS))
def test_load_model(model_path):
    convert_command_and_assert(model_path)


@pytest.mark.parametrize("model_path", get_test_files(TEST_EXECUTE_MODEL_PATHS))
def test_execute_model(model_path):
    execute_command_and_wait(
        model_path, {"optimizationPolicy": "Optimizer Disabled"}, timeout=300
    )
    convert_command_and_assert(model_path)


def test_execute_mnist_df_sharding():
    # Optimizer disabled because these tests are run with tracy build in CI, and tracy
    # build doesn't have op_model (optimizer) support.
    execute_command_and_wait(
        MNIST_SHARDING_PATH,
        {"optimizationPolicy": "Optimizer Disabled"},
        timeout=300,
    )
    convert_command_and_assert(MNIST_SHARDING_PATH)


def test_load_stablehlo_model():
    convert_command_and_assert(MNIST_STABLEHLO_PATH)


def test_execute_mnist_stablehlo():
    execute_command_and_wait(
        MNIST_STABLEHLO_PATH,
        {"optimizationPolicy": "Optimizer Disabled"},
        timeout=300,
    )
    convert_command_and_assert(MNIST_STABLEHLO_PATH)


def test_execute_mnist_with_overrides():
    overrides = {
        'loc("relu_3"("MNISTLinear":4294967295:6))': {
            "named_location": "relu_3",
            "attributes": [
                {"key": "data_type", "value": "f32"},
                {"key": "memory_layout", "value": "tile"},
                {"key": "buffer_type", "value": "dram"},
                {"key": "tensor_memory_layout", "value": "interleaved"},
            ],
        }
    }
    execute_command_and_wait(
        MNIST_SHARDING_PATH,
        {"optimizationPolicy": "Optimizer Disabled", "overrides": overrides},
        timeout=300,
    )
    convert_command_and_assert(MNIST_SHARDING_PATH)


def test_execute_and_check_perf_data_exists():
    print("\n" + "=" * 80)
    print("DEBUG: test_execute_and_check_perf_data_exists - Starting")
    print("=" * 80)

    # Check if expected artifacts dir exists before execution
    tt_mlir_home = os.environ.get("TT_MLIR_HOME", "")
    if tt_mlir_home:
        artifacts_dir = os.path.join(tt_mlir_home, "ttrt-artifacts")
        print(f"DEBUG: Pre-execution check - artifacts_dir: {artifacts_dir}")
        print(f"DEBUG: Pre-execution check - artifacts_dir exists: {os.path.exists(artifacts_dir)}")

    execute_command_and_wait(
        MNIST_SHARDING_PATH,
        {"optimizationPolicy": "Optimizer Disabled"},
        timeout=300,
    )
    print("DEBUG: execute_command_and_wait completed successfully")

    # Check if artifacts were created after execution
    if tt_mlir_home:
        artifacts_dir = os.path.join(tt_mlir_home, "ttrt-artifacts")
        print(f"DEBUG: Post-execution check - artifacts_dir exists: {os.path.exists(artifacts_dir)}")
        if os.path.exists(artifacts_dir):
            try:
                contents = os.listdir(artifacts_dir)
                print(f"DEBUG: Post-execution - artifacts_dir contents: {contents}")
                # Check for perf directory and ops_perf_results.csv
                for item in contents:
                    item_path = os.path.join(artifacts_dir, item)
                    if os.path.isdir(item_path):
                        perf_dir = os.path.join(item_path, "perf")
                        if os.path.exists(perf_dir):
                            print(f"DEBUG: Found perf directory: {perf_dir}")
                            perf_contents = os.listdir(perf_dir)
                            print(f"DEBUG: Perf directory contents: {perf_contents}")
                            csv_file = os.path.join(perf_dir, "ops_perf_results.csv")
                            if os.path.exists(csv_file):
                                print(f"DEBUG: ops_perf_results.csv EXISTS at {csv_file}")
                                # Check file size
                                size = os.path.getsize(csv_file)
                                print(f"DEBUG: ops_perf_results.csv size: {size} bytes")
                            else:
                                print(f"DEBUG: ops_perf_results.csv NOT FOUND at {csv_file}")
            except Exception as e:
                print(f"DEBUG: Error checking artifacts: {e}")

    # First, let's check the model_state on the server side
    print(f"DEBUG: About to call convert_command_and_assert")
    print(f"DEBUG: MNIST_SHARDING_PATH = {MNIST_SHARDING_PATH}")
    
    result = convert_command_and_assert(MNIST_SHARDING_PATH)
    print(f"DEBUG: convert_command_and_assert completed successfully")
    
    # Read the debug log file created by tt_adapter
    log_file = "/tmp/tt_adapter_convert_debug.log"
    if os.path.exists(log_file):
        print(f"\nDEBUG: Contents of {log_file}:")
        with open(log_file, "r") as f:
            print(f.read())
    else:
        print(f"DEBUG: Log file {log_file} does not exist")
    print(f"DEBUG: result type = {type(result)}")
    print(f"DEBUG: result keys = {result.keys() if isinstance(result, dict) else 'N/A'}")
    
    # Print the entire result for debugging
    import json
    print(f"DEBUG: Full result JSON (first 2000 chars):")
    result_str = json.dumps(result, indent=2, default=str)
    print(result_str[:2000])

    if isinstance(result, dict) and "graphs" in result:
        print(f"DEBUG: Number of graphs = {len(result['graphs'])}")
        if len(result["graphs"]) > 0:
            graph = result["graphs"][0]
            print(f"DEBUG: graph[0] keys = {graph.keys()}")
            if "overlays" in graph:
                overlays = graph["overlays"]
                print(f"DEBUG: overlays type = {type(overlays)}")
                print(f"DEBUG: overlays keys = {overlays.keys() if isinstance(overlays, dict) else overlays}")
                print(f"DEBUG: 'perf_data' in overlays = {'perf_data' in overlays}")

                # Print all overlay keys and their types for debugging
                if isinstance(overlays, dict):
                    for key, value in overlays.items():
                        print(
                            f"DEBUG: overlay['{key}'] = {type(value)} (length={len(value) if hasattr(value, '__len__') else 'N/A'})"
                        )
            else:
                print("DEBUG: ERROR - No 'overlays' key in graph[0]")
                print("DEBUG: This means tt_adapter did not add ANY overlays to the graph")
                print("DEBUG: Possible causes:")
                print("DEBUG:   1. perf_node_data is empty (no CSV file found or parsed)")
                print("DEBUG:   2. memory_node_data is empty (memory profiling failed)")
                print("DEBUG:   3. accuracy_node_data is empty (no accuracy data)")
                print("DEBUG:   4. Build configuration issue (explorer not enabled properly)")
        else:
            print("DEBUG: ERROR - graphs list is empty")
    else:
        print(f"DEBUG: ERROR - result is not a dict or has no 'graphs' key")
        print(f"DEBUG: result = {result}")

    print("=" * 80)
    assert "perf_data" in result["graphs"][0]["overlays"]


def test_execute_model_invalid_policy():
    with pytest.raises(AssertionError):
        execute_command_and_wait(
            TEST_EXECUTE_MODEL_PATHS[0],
            {"optimizationPolicy": "Invalid Policy"},
            timeout=300,
        )


def test_execute_and_check_memory_data_exists():
    print("\n" + "=" * 80)
    print("DEBUG: test_execute_and_check_memory_data_exists - Starting")
    print("=" * 80)

    execute_command_and_wait(
        MNIST_SHARDING_PATH,
        {"optimizationPolicy": "Optimizer Disabled"},
        timeout=300,
    )
    print("DEBUG: execute_command_and_wait completed successfully")

    result = convert_command_and_assert(MNIST_SHARDING_PATH)
    print(f"DEBUG: convert_command_and_assert completed successfully")

    # Check what's in the result
    if isinstance(result, dict) and "graphs" in result and len(result["graphs"]) > 0:
        graph = result["graphs"][0]
        if "overlays" in graph:
            overlays = graph["overlays"]
            print(f"DEBUG: overlays keys = {overlays.keys() if isinstance(overlays, dict) else overlays}")

    result_str = str(result)
    print(f"DEBUG: 'display_type' in str(result) = {'display_type' in result_str}")
    print(f"DEBUG: str(result) length = {len(result_str)}")
    print("=" * 80)

    assert "display_type" in str(result)


def test_get_emitc_cpp_code():
    execute_command_and_wait(
        MNIST_SHARDING_PATH,
        {
            "optimizationPolicy": "Optimizer Disabled",
            "generateCppCode": True,
        },
        timeout=300,
    )
    result = convert_command_and_assert(MNIST_SHARDING_PATH)
    assert "cppCode" in result["graphs"][0]


# TODO: figure out if this should be deleted, or adapted with new tests
@pytest.mark.skip(
    "This is now handled by tests under `test/python/golden/test_ttir_models.py`"
)
def test_execute_and_check_accuracy_data_exists():
    # Get the test_mnist path
    test_mnist_path = GET_TTNN_TEST()

    assert (
        test_mnist_path is not None
    ), "Couldn't find test_mnist.ttnn in GENERATED_TTNN_TEST_DIRS"
    execute_command_and_wait(
        test_mnist_path, {"optimizationPolicy": "Optimizer Disabled"}, timeout=300
    )
    result = convert_command_and_assert(test_mnist_path)
    assert "accuracy_data" in result["graphs"][0]["overlays"]
