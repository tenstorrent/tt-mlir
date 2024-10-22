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
CONVERT_URL = "http://" + HOST + ":" + str(PORT) + "/apipost/v1/send_command"
TEST_LOAD_MODEL_PATHS = [
    "test/ttmlir/Dialect/TTNN/mnist_sharding.mlir",
    "tools/explorer/test/*.mlir",
]


def get_test_files():
    files = []
    for path in TEST_LOAD_MODEL_PATHS:
        files.extend(glob.glob(path))
    return files


@pytest.fixture(scope="session", autouse=True)
def start_server(request):
    server_thread = multiprocessing.Process(
        target=model_explorer.visualize,
        kwargs={"extensions": ["tt_adapter"], "host": HOST, "port": PORT},
    )
    server_thread.start()
    time.sleep(1)

    request.addfinalizer(lambda: server_thread.terminate())


@pytest.mark.parametrize("model_path", get_test_files())
def test_load_model(model_path):
    cmd = {
        "extensionId": "tt_adapter",
        "cmdId": "convert",
        "modelPath": model_path,
        "deleteAfterConversion": False,
        "settings": {},
    }

    result = requests.post(CONVERT_URL, json=cmd)
    assert result.ok
    if "error" in result.json():
        print(result.json())
        assert False
