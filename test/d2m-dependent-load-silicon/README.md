# Dependent-load silicon repro

Not a lit test — a manual reproduction of the n150 run recorded in
`docs/src/d2m-dependent-loads.md`. Nothing in TTIR emits a dependent load yet, so
the fixture is built by injecting one into pipeline-generated IR.

`inject.py` takes the pre-split D2M IR for `two_in.mlir` and adds an i32 index
buffer as a second `ins` operand of the weights tilize generic, scalar-reads one
i32 out of its L1 CB, and uses that value as the *row* the weights transfer
reads. With `ttrt run --init arange` the mapping is fully predictable, so the
output names which weights row each core actually fetched.

```bash
source env/activate
ttrt query --save-artifacts
DESC=$(pwd)/ttrt-artifacts/system_desc.ttsys
PL="--ttir-to-ttmetal-pipeline=system-desc-path=$DESC default-input-memspace=dram default-output-memspace=dram"
T=test/d2m-dependent-load-silicon

# 1. Capture pre-split IR at module scope (threading off so module scope prints).
ttmlir-opt "$PL" --mlir-print-ir-before=d2m-split-unified-thread \
  --mlir-print-ir-module-scope --mlir-disable-threading $T/two_in.mlir \
  > /dev/null 2> dump.txt
python3 -c "s=open('dump.txt').read(); i=s.index('IR Dump Before D2MSplitUnifiedThread'); \
  j=s.index(chr(10), s.index('//-----', i+40))+1; open('presplit.mlir','w').write(s[j:])"

# 2. Inject the dependent load.
python3 $T/inject.py presplit.mlir dep.mlir

# 3. Run the rest of the pipeline and serialize.
INNER="d2m-split-unified-thread,d2m-preallocate-mcast-semaphores,d2m-schedule-dma,canonicalize,\
d2m-insert-scalar-access-cb,d2m-lower-load-store-ops-to-dma,d2m-optimize-dma,\
d2m-expand-dma-read-composite-view,d2m-lower-dma-to-fully-indexed-form,\
d2m-normalize-thread-args,d2m-generic-regions-to-funcs,func.func(canonicalize,lower-affine)"
TAIL="func.func(convert-d2m-to-ttkernel,canonicalize,ttkernel-control-dst-section,canonicalize),\
convert-d2m-to-ttmetal,func.func(ttkernel-hoist-inits,ttkernel-dedup-inits),\
func.func(convert-ttkernel-to-emitc,canonicalize,remove-dead-emitc-expressions,form-expressions)"
ttmlir-opt --pass-pipeline="builtin.module(ttcore.device_module(builtin.module($INNER,$TAIL)))" \
  -o dep_final.mlir dep.mlir
ttmlir-translate --ttmetal-to-flatbuffer -o dep.ttm dep_final.mlir

# 4. Run and check.
ttrt run --init arange --save-artifacts dep.ttm
```

Then, with `d = ttrt-artifacts/dep.ttm/run/program_0/`:

```python
import torch
w = torch.load(d + "input_0.pt")[0]
o = torch.load(d + "device_output_0.pt")[0]
# out tile-row i must come from weights tile-row 7-i; 4096*r fingerprints row r.
for i in range(8):
    assert round(o[32 * i, 0].item() / 4096) == 7 - i, i
```

## Loop variants

`inject_loop.py` takes the same pre-split IR and a variant name, putting the
scalar read inside an `scf.for` so both wait/pop cadences run on device:

```bash
python3 $T/inject_loop.py presplit.mlir loop_refill.mlir  refill
python3 $T/inject_loop.py presplit.mlir loop_hoisted.mlir hoisted
```

Then the same steps 3 and 4. `refill` puts the index transfer inside the loop
(pair balances per iteration; the CB has 2 pages and the loop runs 4 times, so a
missing pop hangs on the third `reserve_back`); `hoisted` leaves it outside (pair
brackets the loop). Both predict the same reversal, so the check in step 4 is
unchanged. Run under `timeout` — a deadlock shows up as exit 124.

## Notes

The `7 - (perm / 4096)` formula in `inject.py` gives a reversal. Swapping it for
`(perm / 4096 + 3) % 8` — done in i32 before the `index_cast`, since `arith.remui`
on `index` is not legalized for EmitC — gives a rotation, and running both is what
shows the loaded value determines the address rather than the mapping being baked
into the fixture.

Expect a full-tensor max absolute error of ~15 (0.09% relative). That is the
TTMetal path's math fidelity on this input, not the dependent load: the unmodified
`abs` baseline shows the same error.
