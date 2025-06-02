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

HOST = "localhost"
# Use portpicker to pick a port for us. (say that 10 times fast)
PORT = portpicker.pick_unused_port()
COMMAND_URL = "http://" + HOST + ":" + str(PORT) + "/apipost/v1/send_command"
TEST_LOAD_MODEL_PATHS = [
    "test/ttmlir/Explorer/**/*.mlir",
    "test/ttmlir/Silicon/TTNN/n150/perf/**/*.mlir",
]
MNIST_SHARDING_PATH = "test/ttmlir/Silicon/TTNN/n150/optimizer/mnist_sharding.mlir"
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


def test_execute_mnist_l1_interleaved():
    execute_command_and_wait(
        MNIST_SHARDING_PATH,
        {"optimizationPolicy": "Greedy L1 Interleaved"},
        timeout=300,
    )
    convert_command_and_assert(MNIST_SHARDING_PATH)


def test_execute_mnist_df_sharding():
    execute_command_and_wait(
        MNIST_SHARDING_PATH,
        {"optimizationPolicy": "DF Sharding"},
        timeout=300,
    )
    convert_command_and_assert(MNIST_SHARDING_PATH)


def test_execute_mnist_with_overrides():
    overrides = {
        'loc("matmul_1"("MNISTLinear":4294967295:10))__17': {
            "named_location": "matmul_1",
            "attributes": [
                {"key": "data_type", "value": "f32"},
                {"key": "memory_layout", "value": "tile"},
                {"key": "buffer_type", "value": "dram"},
                {"key": "tensor_memory_layout", "value": "interleaved"},
                {"key": "grid_shape", "value": "[8x8]"},
            ],
        }
    }
    execute_command_and_wait(
        MNIST_SHARDING_PATH,
        {"optimizationPolicy": "DF Sharding", "overrides": overrides},
        timeout=300,
    )
    convert_command_and_assert(MNIST_SHARDING_PATH)


def test_execute_and_check_perf_data_exists():
    execute_command_and_wait(
        MNIST_SHARDING_PATH,
        {"optimizationPolicy": "DF Sharding"},
        timeout=300,
    )
    result = convert_command_and_assert(MNIST_SHARDING_PATH)
    assert "perf_data" in result["graphs"][0]["overlays"]


def test_execute_model_invalid_policy():
    with pytest.raises(AssertionError):
        execute_command_and_wait(
            TEST_EXECUTE_MODEL_PATHS[0],
            {"optimizationPolicy": "Invalid Policy"},
            timeout=300,
        )


def test_execute_and_check_memory_data_exists():
    execute_command_and_wait(
        MNIST_SHARDING_PATH,
        {"optimizationPolicy": "Optimizer Disabled"},
        timeout=300,
    )
    result = convert_command_and_assert(MNIST_SHARDING_PATH)
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
