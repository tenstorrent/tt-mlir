# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

from multichip_overlap_benchmark import _device_kernel_ns, _summarize_tracy
from llama_down_projection_workload import WorkloadConfig, golden, make_operands


def test_down_projection_tensor_parallel_partition_matches_dense_matmul():
    config = WorkloadConfig(m=128, k=14336, n=256, grid_y=2, grid_x=2)
    activations, weight = make_operands(config)
    segment_size = config.k_segment_elements
    partials = []
    for device in range(2):
        partial = torch.zeros((config.m, config.n), dtype=torch.float32)
        for local_segment in range(config.k_segments // 2):
            segment = device * config.k_segments // 2 + local_segment
            start = segment * segment_size
            stop = start + segment_size
            partial += activations[:, start:stop].float() @ weight[start:stop].float()
        partials.append(partial)
    partial_sum = partials[0] + partials[1]
    expected = golden(activations, weight).float()
    pcc = torch.corrcoef(torch.stack((partial_sum.flatten(), expected.flatten())))[
        0, 1
    ].item()

    assert tuple(activations.shape) == (config.m, config.k)
    assert tuple(weight.shape) == (config.k, config.n)
    assert tuple(partial_sum.shape) == (config.m, config.n)
    assert pcc > 0.999


def test_device_kernel_ns_excludes_warmup(tmp_path):
    profile_csv = tmp_path / "profile_log_device.csv"
    profile_csv.write_text(
        "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64\n"
        "PCIe slot,core_x,core_y,RISC processor type,timer_id,"
        "time[cycles since reset],data,run host ID,trace id,trace id counter,"
        "zone name,type,source line,source file,meta data\n"
        "0,1,1,TRISC,1,100,0,7,,,TRISC-KERNEL,ZONE_START,0,,\n"
        "0,1,1,TRISC,1,200,0,7,,,TRISC-KERNEL,ZONE_END,0,,\n"
        "0,1,1,TRISC,1,1000,0,7,,,TRISC-KERNEL,ZONE_START,0,,\n"
        "0,1,1,TRISC,1,1130,0,7,,,TRISC-KERNEL,ZONE_END,0,,\n"
    )

    assert _device_kernel_ns(profile_csv, warmup=1, iterations=1) == 130


def test_tracy_summary_aggregates_zones_per_submit(tmp_path, capsys):
    tracy_csv = tmp_path / "trace.csv"
    tracy_csv.write_text(
        "name,src_file,src_line,zone_name,zone_text,ns_since_start,"
        "exec_time_ns,thread,special_parent_text\n"
        "submit,,, , ,100,1000,1,\n"
        "MeshShardCommand,,, , ,150,100,1,\n"
        "MeshShardCommand,,, , ,300,200,1,\n"
        "submit,,, , ,1200,2000,1,\n"
        "MeshShardCommand,,, , ,1300,400,1,\n"
        "MeshShardCommand,,, , ,1800,600,1,\n"
    )

    _summarize_tracy(tracy_csv, measured_iterations=1)

    output = capsys.readouterr().out
    assert "tracy submit: count=1 mean_us=2.0" in output
    assert "MeshShardCommand: count_per_submit=2 mean_total_us=1.0" in output
