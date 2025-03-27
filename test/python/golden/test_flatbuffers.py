# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess


def test_flatbuffers(artifact_path: str):
    subprocess.run(
        f"ttrt run {artifact_path}/ttnn", shell=True, capture_output=True, text=True
    )
