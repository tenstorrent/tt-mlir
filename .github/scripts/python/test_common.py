#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import hashlib


def compute_hash(test):
    machine = test.get("machine", "")
    image = test.get("image", "")
    test_type = test.get("type", "")
    path = test.get("path", "")
    args = test.get("args", "")
    flags = test.get("flags", "")
    hash_string = f"{machine}-{image}-{test_type}-{path}-{args}-{flags}"
    hash = hashlib.md5(hash_string.encode()).hexdigest()
    return hash, hash_string
