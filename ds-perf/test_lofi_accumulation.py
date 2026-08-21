"""Does the LoFi error accumulate across a decode MLP stack, or stay bounded?

Per-op PCC cannot answer this: a per-op residual of 1e-4 could compound through 16
layers or could cancel.  So run the whole stack on device at each fidelity and compare
against an fp32 torch chain of the same weights.

Stack is llama_3_2_1b's MLP shape (hidden 2048, intermediate 8192, 16 layers), with
distinct bfp8 weights per layer, RMSNorm, SiLU gating and a residual add -- so errors
have a real path to compound.  Reports PCC and relative error along the way.

The RMSNorm is load-bearing, not decoration: without it the residual stream grows
unbounded and both fidelities overflow bf16 to NaN by layer 13, which measures the
harness rather than the arithmetic.  Real stacks renormalize every layer.

Caveat: plain interleaved matmuls, not the DRAM-sharded program config.  Fidelity drives
the same MVMUL sequence either way, so this isolates fidelity; it is not a perf test.
"""
import pytest, torch, ttnn

H, I, LAYERS = 2048, 8192, 16


def _pcc(x, y):
    x = x.flatten().to(torch.float64); y = y.flatten().to(torch.float64)
    xm, ym = x - x.mean(), y - y.mean()
    d = xm.norm() * ym.norm()
    return float("nan") if d == 0 else float((xm @ ym) / d)


# bf16 weights isolate the fidelity effect from bfp8 quantization: if LoFi-bf16 tracks
# HiFi2-bf16 closely, the accumulated error under bfp8 is quantization, not fidelity.
@pytest.mark.parametrize("wdt_name,wdt", [("bfp8", ttnn.bfloat8_b), ("bf16", ttnn.bfloat16)],
                         ids=["bfp8", "bf16"])
@pytest.mark.parametrize("fid_name,fid", [("LoFi", ttnn.MathFidelity.LoFi),
                                          ("HiFi2", ttnn.MathFidelity.HiFi2)],
                         ids=["LoFi", "HiFi2"])
def test_accumulation(device, fid_name, fid, wdt_name, wdt):
    torch.manual_seed(0)
    x0 = torch.randn(1, 1, 32, H).bfloat16()
    Wg = [torch.randn(1, 1, H, I).bfloat16() * (H ** -0.5) for _ in range(LAYERS)]
    Wu = [torch.randn(1, 1, H, I).bfloat16() * (H ** -0.5) for _ in range(LAYERS)]
    Wd = [torch.randn(1, 1, I, H).bfloat16() * (I ** -0.5) for _ in range(LAYERS)]

    # fp32 reference chain
    def rms(t):
        return t / (t.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()

    ref = x0.to(torch.float32)
    refs = []
    for l in range(LAYERS):
        n = rms(ref)
        g = n @ Wg[l].to(torch.float32)
        u = n @ Wu[l].to(torch.float32)
        ref = ref + (torch.nn.functional.silu(g) * u) @ Wd[l].to(torch.float32)
        refs.append(ref.clone())

    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, packer_l1_acc=True)
    dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT,
                                        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = dev(x0, ttnn.bfloat16)
    print(f"\n[{fid_name}/{wdt_name}] {LAYERS}-layer MLP stack, hidden={H} inter={I}")
    print(f"  {'layer':>5s} {'PCC':>12s} {'mean rel err':>13s} {'max rel err':>12s}")
    for l in range(LAYERS):
        wg, wu, wd = dev(Wg[l], wdt), dev(Wu[l], wdt), dev(Wd[l], wdt)
        n = ttnn.rms_norm(x, epsilon=1e-6)
        g = ttnn.linear(n, wg, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        u = ttnn.linear(n, wu, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        h = ttnn.multiply(ttnn.silu(g), u)
        y = ttnn.linear(h, wd, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        x = ttnn.add(x, y)
        ttnn.synchronize_device(device)
        for t in (n, g, u, h, y, wg, wu, wd):
            ttnn.deallocate(t)
        if l == 0 or l == LAYERS - 1:
            got = ttnn.to_torch(x).to(torch.float32)
            r = refs[l]
            print(f"  {l+1:>5d} {_pcc(r, got):>12.7f} "
                  f"{float((got-r).abs().mean()/r.abs().mean()):>13.6f} "
                  f"{float((got-r).abs().max()/r.abs().max()):>12.6f}")
    ttnn.deallocate(x)
