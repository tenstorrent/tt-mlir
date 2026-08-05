# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from multichip_overlap_benchmark import _device_kernel_ns


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
