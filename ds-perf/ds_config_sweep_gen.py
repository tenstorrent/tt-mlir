"""Generate single-op DS matmul graphs for fleet shapes, sweeping the free variables:
C = in0 activation shard core count (C | kTiles), w = in0_block_w (w | kTiles/C).
Banks stay at 8 -- the weight is always DRAM width-sharded over all 8.
Mirrors the harness that reproduced the full-compile endpoints to <0.5%."""
import argparse, json
from pathlib import Path

GRID_W = 11          # worker grid width used for row-wrapped in0/out placements
BANKS  = 8

def ranges(n, width=GRID_W):
    """Row-wrapped core_range_set covering exactly n cores."""
    full, rem = divmod(n, width)
    parts = []
    if full: parts.append(f"#ttnn.core_range<(0,0), ({width-1},{full-1})>")
    if rem:  parts.append(f"#ttnn.core_range<(0,{full}), ({rem-1},{full})>")
    return "core_ranges = <[" + ", ".join(parts) + "]>"

def emit(name, K, N, C, w, per_core_n, dt, preamble, device_line, cfg_kind="ds"):
    kT, nT = K//32, -(-N//32)
    sn = -(-nT//BANKS)
    wt = {"bfp8":"bfp_bf8","bfp4":"bfp_bf4"}[dt]
    kpc = kT//C
    nout = nT                       # output width-sharded one tile per core
    L = f"""
#a_host = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x{kT}x!ttcore.tile<32x32, bf16>, #system_memory>>
#a_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x{kT}x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#in0    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x{C}>, memref<1x{kpc}x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, {ranges(C)}>
#w_host = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<{kT}x{nT}x!ttcore.tile<32x32, bf16>, #system_memory>>
#w_h8   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<{kT}x{nT}x!ttcore.tile<32x32, {wt}>, #system_memory>>
#w_il   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<{kT}x{nT}x!ttcore.tile<32x32, {wt}>, #dram>, <interleaved>>
#w_ws   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x{BANKS}>, memref<{kT}x{sn}x!ttcore.tile<32x32, {wt}>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), ({BANKS-1},0)>]>>
#out    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x{nout}>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, {ranges(nout)}>
"""
    AH=f"tensor<32x{K}xbf16, #a_host>"; AD=f"tensor<32x{K}xbf16, #a_dram>"
    A0=f"tensor<32x{K}xbf16, #in0>"
    WH=f"tensor<{K}x{N}xbf16, #w_host>"; W8=f"tensor<{K}x{N}x!ttcore.tile<32x32, {wt}>, #w_h8>"
    WI=f"tensor<{K}x{N}x!ttcore.tile<32x32, {wt}>, #w_il>"
    WS=f"tensor<{K}x{N}x!ttcore.tile<32x32, {wt}>, #w_ws>"
    OU=f"tensor<32x{N}xbf16, #out>"
    COMPUTE=("#ttnn.device_compute_kernel_config<math_fidelity = hifi2, "
             "math_approx_mode = false, fp32_dest_acc_en = false, "
             "packer_l1_acc = true, dst_full_sync_en = false>")
    cfg=("#ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<"
         f"in0_block_w = {w}, per_core_m = 1, per_core_n = {per_core_n}>")
    return f"""{preamble}
{L}
module @{name} attributes {{ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>}} {{
  ttcore.device_module {{
    builtin.module @{name} attributes {{ttcore.system_desc = #system_desc}} {{
{device_line}
      func.func @main(%argA: {AH}, %argW: {WH}) -> {OU} attributes {{tt.function_type = "forward_device"}} {{
        %dev = "ttnn.get_device"() <{{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}}> : () -> !ttnn.device
        %w8 = "ttnn.typecast"(%argW) : ({WH}) -> {W8}
        %wd = "ttnn.to_device"(%w8, %dev) : ({W8}, !ttnn.device) -> {WI}
        %w = "ttnn.to_memory_config"(%wd) : ({WI}) -> {WS}
        %ad = "ttnn.to_device"(%argA, %dev) : ({AH}, !ttnn.device) -> {AD}
        %a = "ttnn.to_memory_config"(%ad) : ({AD}) -> {A0}
        %o = "ttnn.matmul"(%a, %w) <{{
          compute_config = {COMPUTE},
          matmul_program_config = {cfg},
          transpose_a = false, transpose_b = false}}>
          {{ttcore.weight_dtype = "{wt}"}} : ({A0}, {WS}) -> {OU}
        return %o : {OU}
      }}
    }}
  }}
}}
"""

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True); ap.add_argument("--outdir",required=True)
    ap.add_argument("--plan",required=True)
    a=ap.parse_args()
    lines=Path(a.source).read_text().splitlines()
    mod=next(i for i,l in enumerate(lines) if l.startswith("module @"))
    pre="\n".join(lines[:mod])
    dev=next(l for l in lines if "ttcore.device @default_device" in l)
    od=Path(a.outdir); od.mkdir(parents=True,exist_ok=True)
    n=0
    for p in json.load(open(a.plan)):
        txt=emit(p["name"],p["K"],p["N"],p["C"],p["w"],p["pcn"],p["dt"],pre,dev)
        (od/f"{p['name']}.mlir").write_text(txt); n+=1
    print(f"wrote {n} graphs to {od}")
