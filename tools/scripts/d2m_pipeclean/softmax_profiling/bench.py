# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json, time, torch
import ttrt.binary as binary
import ttrt.runtime as rt

DT = {
    "Float32": rt.DataType.Float32,
    "BFloat16": rt.DataType.BFloat16,
    "UInt32": rt.DataType.UInt32,
}
TDT = {"Float32": torch.float32, "BFloat16": torch.bfloat16, "UInt32": torch.int32}


def bench(name, fb, loops=1000, warmup=30):
    fbb = binary.load_binary_from_path(fb)
    rt.set_compatible_device_runtime(fbb)
    opts = rt.MeshDeviceOptions()
    opts.mesh_shape = list(fbb.get_program_mesh_shape(0))
    dev = rt.open_mesh_device(opts)
    try:
        specs = json.loads(fbb.get_program_inputs_as_json(0))
        ins = []
        for i, s in enumerate(specs):
            d = s["desc"]
            shape = [int(x) for x in d["shape"]]
            dtype = d["layout"]["memory_desc"]["data_type"]
            tdt = TDT[dtype]
            t = (
                torch.ones(shape, dtype=torch.int32)
                if dtype == "UInt32"
                else torch.randn(shape).to(tdt)
            ).contiguous()
            ht = rt.create_borrowed_host_tensor(
                t.data_ptr(),
                list(t.shape),
                list(t.stride()),
                t.element_size(),
                DT[dtype],
            )
            ins.append(rt.to_layout(ht, dev, rt.get_layout(fbb, 0, i), True))

        def one():
            o = rt.submit(dev, fbb, 0, ins)
            rt.wait(o)
            for x in o:
                rt.deallocate_tensor(x, force=True)

        for _ in range(warmup):
            one()
        t0 = time.perf_counter()
        for _ in range(loops):
            one()
        us = (time.perf_counter() - t0) / loops * 1e6
        print(f"RESULT {name:14} {us:8.1f}")
    finally:
        rt.close_mesh_device(dev)


import os

# Flatbuffers are produced by gen_cases.sh into <repo-root>/_fb (override with FB_DIR).
FB = os.environ.get("FB_DIR", "_fb")
for name, fb in [
    ("d2m-fused 1x1", "d2m_fused_1x1.ttm"),
    ("ttnn 1x1", "ttnn_softmax_1x1.ttnn"),
    ("d2m-fused 2x2", "d2m_fused_2x2.ttm"),
    ("ttnn 2x2", "ttnn_softmax_2x2.ttnn"),
    ("d2m-fused 3x3", "d2m_fused_3x3.ttm"),
    ("ttnn 3x3", "ttnn_softmax_3x3.ttnn"),
]:
    try:
        bench(name, os.path.join(FB, fb))
    except Exception as e:
        print(f"RESULT {name:14} FAIL {str(e).splitlines()[0][:70]}")
