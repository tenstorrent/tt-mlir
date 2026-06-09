# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json, time, sys, torch
import ttrt.binary as binary
import ttrt.runtime as rt

DT = {"Float32": rt.DataType.Float32}
TDT = {"Float32": torch.float32}
fb = sys.argv[1]
S = int(sys.argv[2])
fbb = binary.load_binary_from_path(fb)
rt.set_compatible_device_runtime(fbb)
opts = rt.MeshDeviceOptions()
opts.mesh_shape = list(fbb.get_program_mesh_shape(0))
dev = rt.open_mesh_device(opts)
try:
    torch.manual_seed(0)
    a = torch.randn(S, S)
    specs = json.loads(fbb.get_program_inputs_as_json(0))
    ins = []
    for i, s in enumerate(specs):
        d = s["desc"]
        shape = [int(x) for x in d["shape"]]
        dtype = d["layout"]["memory_desc"]["data_type"]
        t = (
            (
                a if shape == [S, S] else torch.ones(shape, dtype=torch.int32)
            ).contiguous()
            if dtype == "Float32"
            else torch.ones(shape, dtype=torch.int32).contiguous()
        )
        t = t.to(TDT.get(dtype, torch.int32)).contiguous() if dtype == "Float32" else t
        ht = rt.create_borrowed_host_tensor(
            t.data_ptr(),
            list(t.shape),
            list(t.stride()),
            t.element_size(),
            DT.get(dtype, rt.DataType.UInt32),
        )
        ins.append(rt.to_layout(ht, dev, rt.get_layout(fbb, 0, i), True))
    # PCC
    o = rt.submit(dev, fbb, 0, ins)
    rt.wait(o)
    hv = rt.to_host(o[0], untilize=True)[0]
    out = torch.empty(S, S, dtype=torch.float32)
    rt.memcpy(out.data_ptr() if hasattr(out, "data_ptr") else out, hv)
    for x in o:
        rt.deallocate_tensor(x, force=True)
    ref = torch.softmax(a, dim=1)
    x = ref.flatten().double()
    y = out.flatten().double()
    pcc = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
    # bench
    def one():
        oo = rt.submit(dev, fbb, 0, ins)
        rt.wait(oo)
        for z in oo:
            rt.deallocate_tensor(z, force=True)

    for _ in range(30):
        one()
    t0 = time.perf_counter()
    for _ in range(1000):
        one()
    us = (time.perf_counter() - t0) / 1000 * 1e6
    print(f"RESULT fused {S}x{S}: PCC={pcc:.6f}  {us:.1f} us/submit")
finally:
    rt.close_mesh_device(dev)
