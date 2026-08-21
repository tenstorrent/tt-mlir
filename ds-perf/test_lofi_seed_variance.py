"""Is the ~5e-5 PCC gap between LoFi and HiFi2 on bfp8 real, or seed noise?
Same shapes, 8 input samples each, paired (identical inputs per fidelity)."""
import pytest, torch, ttnn, statistics as st
TILE, BANKS = 32, 8
SHAPES=[("f7b_down",23040,3072,18),("q3b_gateup",2048,11008,4),("l1b_down",8192,2048,32)]
NSEED=8
def _rect(device,n):
    cg=device.compute_with_storage_grid_size(); full,rem=divmod(n,cg.x); rs=[]
    if full: rs.append(ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(cg.x-1,full-1)))
    if rem:  rs.append(ttnn.CoreRange(ttnn.CoreCoord(0,full),ttnn.CoreCoord(rem-1,full)))
    return ttnn.CoreRangeSet(rs)
def _pcc(x,y):
    x=x.flatten().to(torch.float64); y=y.flatten().to(torch.float64)
    xm,ym=x-x.mean(),y-y.mean(); d=xm.norm()*ym.norm()
    return float("nan") if d==0 else float((xm@ym)/d)
def run(device,K,N,w,fid,seed):
    M=32; nt=-(-N//TILE); sn=-(-nt//BANKS)
    torch.manual_seed(seed)
    a=torch.randn(1,1,M,K).bfloat16(); b=torch.randn(1,1,K,N).bfloat16()
    a_t=ttnn.from_torch(a,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.BufferType.L1,
            ttnn.ShardSpec(_rect(device,BANKS),[M,K//BANKS],ttnn.ShardOrientation.ROW_MAJOR)))
    b_t=ttnn.from_torch(b,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.BufferType.DRAM,
            ttnn.ShardSpec(_rect(device,BANKS),[K,sn*TILE],ttnn.ShardOrientation.ROW_MAJOR)))
    out=ttnn.linear(a_t,b_t,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=w,per_core_M=1,per_core_N=sn,fused_activation=None),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.BufferType.L1,
            ttnn.ShardSpec(_rect(device,BANKS),[M,sn*TILE],ttnn.ShardOrientation.ROW_MAJOR)),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=fid,packer_l1_acc=True))
    ttnn.synchronize_device(device)
    got=ttnn.to_torch(out)[...,:N].to(torch.float32)
    ttnn.deallocate(out); ttnn.deallocate(a_t); ttnn.deallocate(b_t)
    return got, torch.matmul(a.to(torch.float32),b.to(torch.float32))

@pytest.mark.parametrize("shape",SHAPES,ids=[s[0] for s in SHAPES])
def test_seeds(device,shape):
    sid,K,N,w=shape; d=[]
    print(f"\n[{sid}] {K}x{N} bfp8 w={w}, {NSEED} paired samples")
    print(f"  {'seed':>4s} {'LoFi PCC':>11s} {'HiFi2 PCC':>11s} {'delta':>11s}")
    for s in range(NSEED):
        lo,g=run(device,K,N,w,ttnn.MathFidelity.LoFi,s)
        hi,_=run(device,K,N,w,ttnn.MathFidelity.HiFi2,s)
        pl,ph=_pcc(g,lo),_pcc(g,hi); d.append(pl-ph)
        print(f"  {s:>4d} {pl:>11.7f} {ph:>11.7f} {pl-ph:>+11.7f}")
    m,sd=st.mean(d),(st.stdev(d) if len(d)>1 else 0.0)
    print(f"  mean delta {m:+.7f}  stdev {sd:.7f}  "
          f"|mean|/stdev = {abs(m)/sd if sd else float('inf'):.1f}")
    print(f"  HiFi2 better in {sum(1 for x in d if x<0)}/{len(d)} samples")
