# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttrt.common.api import API


def test_perf_deprecated():
    API.initialize_apis()
    perf_instance = API.Perf()
    result_code, results = perf_instance()
    assert result_code == 1, "deprecated perf should return non-zero exit code"
