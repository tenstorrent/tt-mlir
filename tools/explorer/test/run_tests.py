# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import model_explorer

import requests
import time
import multiprocessing
import pytest
import glob

HOST = "localhost"
PORT = 8002
COMMAND_URL = "http://" + HOST + ":" + str(PORT) + "/apipost/v1/send_command"
TEST_LOAD_MODEL_PATHS = [
    "test/ttmlir/Dialect/TTNN/optimizer/mnist_sharding.mlir",
    "tools/explorer/test/models/*.mlir",
]
TEST_EXECUTE_MODEL_PATHS = [
    "test/ttmlir/Silicon/TTNN/optimizer/mnist_sharding_tiled.mlir",
]


def get_test_files(paths):
    files = []
    for path in paths:
        files.extend(glob.glob(path))
    return files


def send_command(command, model_path, settings):
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
            response = send_command("status_check", "", {})
            if response.status_code == 200 and response.json().get("graphs")[0].get(
                "isDone"
            ):
                return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            raise Exception("Status check request failed")
        time.sleep(1)
    raise RuntimeError(
        f"Execution did not finish within {MODEL_EXECUTION_TIMEOUT} seconds"
    )


def execute_command_and_wait(model_path, settings, timeout):
    execute_command(model_path, settings)
    adapter_response = wait_for_execution_to_finish(timeout)
    assert "graphs" in adapter_response
    assert len(adapter_response["graphs"]) == 1
    response = adapter_response["graphs"][0]
    assert response["isDone"]
    assert response["error"] is None


@pytest.fixture(scope="function", autouse=True)
def start_server(request):
    server_thread = multiprocessing.Process(
        target=model_explorer.visualize,
        kwargs={"extensions": ["tt_adapter"], "host": HOST, "port": PORT},
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


@pytest.mark.parametrize("model_path", get_test_files(TEST_LOAD_MODEL_PATHS))
def test_load_model(model_path):
    result = send_command("convert", model_path, {})
    assert result.ok
    if "error" in result.json():
        print(result.json())
        assert False


@pytest.mark.parametrize("model_path", get_test_files(TEST_EXECUTE_MODEL_PATHS))
def test_execute_model(model_path):
    execute_command_and_wait(
        model_path, {"optimizationPolicy": "DF Sharding"}, timeout=60
    )


def test_execute_mnist_l1_interleaved():
    execute_command_and_wait(
        "test/ttmlir/Silicon/TTNN/optimizer/mnist_sharding_tiled.mlir",
        {"optimizationPolicy": "L1 Interleaved"},
        timeout=60,
    )


def test_execute_mnist_optimizer_disabled():
    execute_command_and_wait(
        "test/ttmlir/Silicon/TTNN/optimizer/mnist_sharding_tiled.mlir",
        {"optimizationPolicy": "Optimizer Disabled"},
        timeout=60,
    )


def test_execute_model_invalid_policy():
    with pytest.raises(AssertionError):
        execute_command_and_wait(
            TEST_EXECUTE_MODEL_PATHS[0],
            {"optimizationPolicy": "Invalid Policy"},
            timeout=60,
        )
