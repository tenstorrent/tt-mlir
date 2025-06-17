# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn


class DeviceGetter:
    _instance = None

    def __init__(self):
        raise RuntimeError("This is Singleton, invoke get_instance() instead.")

    @classmethod
    def get_device(cls):
        if cls._instance == None:
            cls._instance = ttnn.open_device(device_id=0)
        return cls._instance
