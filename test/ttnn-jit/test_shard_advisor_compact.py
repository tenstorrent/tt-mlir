# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttnn_jit._src.shard_advisor import ShardAdvisor


def test_compact_mode_disables_full_trace_and_enables_lightweight_stats(tmp_path):
    advisor = ShardAdvisor(
        func=None,
        optimization_level=2,
        out_dir=str(tmp_path),
        decision_trace=False,
    )

    options = advisor._build_options("/tmp/system.ttsys", "/tmp/trace")

    assert "enable-decision-trace=false" in options
    assert "enable-compile-time-stats=true" in options
    assert "decision-trace-dir=" not in options


def test_full_trace_remains_the_api_default(tmp_path):
    advisor = ShardAdvisor(func=None, out_dir=str(tmp_path))

    options = advisor._build_options("/tmp/system.ttsys", "/tmp/trace")

    assert "enable-decision-trace=true" in options
    assert "decision-trace-dir=/tmp/trace" in options
    assert "enable-compile-time-stats=true" not in options
