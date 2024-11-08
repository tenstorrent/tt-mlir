# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import ttrt
import ttrt.runtime
import torch
from ttrt.common.query import Query
from ttrt.common.util import *

TT_MLIR_HOME = os.environ.get("TT_MLIR_HOME", "")


class Helper:
    def __init__(self, logger=None):
        self.artifacts_dir = f"{os.getcwd()}/ttrt-artifacts"
        self.logger = logger if logger is not None else Logger()
        self.logging = self.logger.get_logger()
        self.file_manager = FileManager(self.logger)
        self.artifacts = Artifacts(
            self.logger, self.file_manager, artifacts_folder_path=self.artifacts_dir
        )
        self.query = Query({"--quiet": True}, self.logger, self.artifacts)
        self.query()
        self.test_name = None
        self.binary_path = None
        self.binary = None

    def initialize(self, test_name, binary_path=None):
        self.test_name = test_name
        if binary_path:
            self.binary_path = binary_path
            self.binary = Binary(self.logger, self.file_manager, binary_path)

    def teardown(self):
        self.test_name = None
        self.binary_path = None
        self.binary = None

    def check_constraints(self):
        if not self.binary:
            return
        self.binary.check_version()
        self.binary.check_system_desc(self.query)


class DeviceContext:
    def __init__(self, device_ids):
        self.device = ttrt.runtime.open_device(device_ids)

    def __enter__(self):
        return self.device

    def __exit__(self, exc_type, exc_value, traceback):
        ttrt.runtime.close_device(self.device)


def assert_tensors_match(tensor1, tensor2):
    assert torch.allclose(tensor1, tensor2)


def assert_pcc(x, y, threshold=0.99):
    combined = torch.stack([x.flatten(), y.flatten()])
    pcc = torch.corrcoef(combined)[0, 1].item()
    assert pcc >= threshold, f"Expected pcc {pcc} >= {threshold}"
