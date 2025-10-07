# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# common functions for test scripts

import hashlib


def compute_hash(test, machine, image):
    test_type = test.get("script", "")
    args = test.get("args", "")
    reqs = test.get("reqs", "")
    hash_string = f"{machine}-{image}-{test_type}-{args}-{reqs}"
    hash = hashlib.md5(hash_string.encode()).hexdigest()
    return hash, hash_string
