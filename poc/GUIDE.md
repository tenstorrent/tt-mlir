# X280 PoC — Concepts Guide

A walkthrough of everything you need to know to read the PoC code if you
come from an ML-compiler background and have never written firmware,
programmed a NOC, or thought about TLBs as anything other than a
page-table cache.

The PoC itself lives in this directory; this document is for
understanding *why* the code looks the way it does.

---

## 1. Why this PoC exists

tt-mlir lowers ML graphs to either Tensix (the matmul/eltwise engines on
Tenstorrent chips) or to the host CPU when an op isn't natively supported
on Tensix yet. The latter is called **CPU hoisting** — a tensor lives in
device DRAM, the runtime DMAs it to host RAM, runs an x86 kernel, DMAs the
result back, and the next op continues on Tensix. Both DMAs cross PCIe,
which is where the cost lives: the compute on x86 is often dwarfed by the
hop in and out of host memory.

Blackhole has four **L2CPU** tiles, each containing a SiFive X280 (a
quad-core 64-bit RISC-V with a vector unit, 4 GiB of attached DRAM, and an
L3 cache). The X280 is *on the chip*, sharing the same on-chip network as
Tensix. If we can run hoisted ops on the X280 instead of the host x86, we
keep PCIe out of the inner loop entirely.

This PoC is the smallest convincing demonstration that the X280 can do
that: boot, see a tensor TTNN computed, modify it in place, hand control
back. There's no compiler integration here yet — the goal is to validate
the *substrate* on which compiler integration will eventually rest.

---

## 2. The Blackhole chip in five minutes

A Blackhole chip is a 2D grid of **tiles** connected by a **NOC**
(network-on-chip). Each tile has an `(x, y)` coordinate and a type:

- **Tensix** tiles — the matmul/eltwise compute. ~140 of them on a p150,
  arranged in a roughly 14×10 region (with some columns reserved).
- **DRAM** tiles — sit at NOC0 columns 0 and 9. Each one is the bus to
  one GDDR6 channel. There are 8 GDDR channels; on this card 7 are
  enabled (one is harvested). Each channel exposes 3 NOC ports, so the
  chip has 24 DRAM tiles in total but only 7×3 active.
- **L2CPU** tiles — at column 8, rows 3 / 5 / 7 / 9. Each has a 4-core
  X280, an L3 cache, and 4 GiB of GDDR attached over AXI.
- **ARC** tile — at `(8, 0)`. Microcontroller responsible for power,
  clocks, telemetry, and miscellaneous chip bring-up.
- **PCIe** tile — what the host x86 talks to.
- **Ethernet** tiles — for chip-to-chip links, irrelevant to this PoC.

When tile A wants to read tile B's memory, it sends a NOC packet
`(B.x, B.y, addr, size)` and B's responder hands the bytes back. Same for
writes. Tiles never share DRAM the way x86 cores share L3 — every
inter-tile load/store is a NOC transaction.

"Harvesting" means the chip arrived from the fab with some tiles
disabled (silicon yield mitigation). On this card, 2 Tensix cores, 1 DRAM
slice, and most Ethernet are harvested, so the *active* tile set is
slightly smaller than the geometric grid.

---

## 3. How the X280 sees memory

Every tile has its own physical address space. From inside L2CPU0, the
X280's CPU-visible map looks (roughly) like:

| Range | What |
|---|---|
| `0x0000_0000` – `0x01FF_FFFF` | local peripherals (CLINT, PLIC, L3 controller, NOC TLB config) |
| `0x0200_0000` – `0x02FF_FFFF` | more peripherals (L3 cache regs, L2 prefetchers) |
| `0x0300_0000` – `0x4000_2FFF_FFFF` | **System Port** (uncached) |
| `0x4000_3000_0000` – `0x7FFF_FFFF_FFFF` | **Memory Port** (cacheable) |

The two big ranges are aliases of the same backing fabric — they differ
*only* in what the X280's L1/L2/L3 caches do with accesses.

- **Memory Port** loads/stores go through the X280's caches. Faster on
  re-access, but the cache doesn't snoop NOC traffic, so what the X280
  writes can sit dirty in L3 indefinitely from the perspective of anyone
  outside the tile.
- **System Port** loads/stores bypass the cache entirely. Slower per
  access, but every store hits the bus immediately.

The bottom 4 GiB of *both* ranges maps to the L2CPU's own GDDR (its
"local DRAM"). So `0x4000_3020_0000` and `0x3020_0000` refer to the same
DRAM byte; the difference is whether the X280's L3 sits between you and
that byte. The PoC firmware lives at `0x4000_3000_0000` (Memory Port,
cacheable — instruction fetches *want* to be cached).

The address space goes up to ~64 TiB because Memory Port and System Port
also cover NOC TLB windows (next section).

---

## 4. The NOC, briefly

The NOC is two independent torus-style packet networks (NOC0 and NOC1)
that connect every tile. PoC code only uses NOC0.

A NOC transaction is `(noc_id, x, y, addr, size, direction)`. The X280
issues these implicitly via its NOC TLBs (next section); the host issues
them via UMD's `write_to_device(tt_xy_pair{x, y}, addr, ...)`.

The NOC has no L2-style snooping. If you write `5` to a DRAM bank from
the host, the bank's controller updates DRAM. If a Tensix core then
*reads* that bank into its L1, it gets `5`. But if the same Tensix had
previously cached that address in L1, the host's write doesn't invalidate
the Tensix L1. This is fine in TTNN's normal flow because Tensix kernels
explicitly DMA from DRAM each operation, but it's worth knowing.

For this PoC the NOC matters because it's how the X280 reaches *other
tiles' memory*. The X280 has no AXI connection to Tensix DRAM — every
load and store from the X280 to a tensor in step 4 is a NOC packet.

---

## 5. NOC TLBs — the bridge between X280 addresses and NOC packets

The "TLB" name is an unfortunate collision. These have nothing to do with
virtual memory page tables. A **NOC TLB** is a register-programmable
window: you tell it "addresses in *this* range of my CPU map should be
turned into NOC packets at *that* `(x, y, addr)`," and from then on
ordinary loads/stores in that range produce NOC traffic.

The X280 has two pools of windows:

| Pool | Count | Window size | Use |
|---|---|---|---|
| 2 MiB windows | 224 | 2 MiB each | Fine-grained — point at a small region of one tile |
| 128 GiB windows | 32 | 128 GiB each | Coarse — point at a whole tile's address space |

A 128 GiB window is more than enough to cover an entire 4 GiB DRAM bank
with `addr = 0`, so step 4 uses 128 GiB windows: one per active DRAM
bank, each pointing at the bank's NOC `(x, y)` with `addr = 0`. To touch
offset `O` of bank `B`, the X280 just loads/stores at:

```
window_base + B * 128GiB + O
```

The window-base address differs between System Port and Memory Port
aliases. Step 4 uses `0x80430000000` (System Port) so its tensor stores
land in DRAM directly without sitting in the X280's L3.

Programming a 128 GiB window is three uint32 writes to a control
register at `0x2FF00E00 + N * 0x0c`, where `N` is the window index. The
encoding is documented in `tt-bh-linux/docs/addressing.md`; for the
unicast case the PoC needs, only `x_end` and `y_end` (the destination
coords) matter, the rest stay zero.

---

## 6. Bare-metal firmware basics

The X280 firmware (`fw/*.c`) is **bare-metal**: no operating system, no
libc, no malloc, no stdio. It runs from physical address `0x4000_3000_0000`
(the start of L2CPU0 DRAM) and exits only via `wfi` (the RISC-V
"wait-for-interrupt" instruction, which here just means "park").

You write bare-metal firmware as if you were writing a 1990s embedded
program:

- `__asm__ volatile(...)` for any instruction the compiler doesn't know
  about (memory fences, CSR writes, custom CMO opcodes).
- A linker script (`fw.ld`) that pins `.text` to a specific address and
  sets up the stack.
- A startup file (`fw_start.S`) that runs *before* `fw_main` and sets up
  whatever the C code expects (stack pointer, FP unit enable, parking
  the harts you don't want).
- Cross-compilation (`clang-20 --target=riscv64-unknown-elf`) because
  your build host is x86_64.

A subtle thing: on RISC-V, the FP unit is *off* at reset (`mstatus.FS =
00`). The first FP instruction traps with illegal-instruction unless
you've set `FS` to a non-zero value. The PoC's `fw_start.S` sets `FS = 11`
(Dirty) so any later `flw` / `fadd.s` / `fmv.*` works. This is one of
those quirks you only learn by hitting it: see the README's "Silicon
quirks" entry.

The four harts of the L2CPU all enter `_start` at reset (because the
boot ROM programs all four reset vectors to the same address). The PoC
parks harts 1-3 in `wfi` and runs everything on hart 0.

---

## 7. UMD — how the host pokes the chip

UMD ("User Mode Driver") is the C++ library that gives the host process
access to the chip without going through tt-metal's higher-level
abstractions. It's what tt-metal itself uses underneath. For this PoC, we
use UMD directly because we're driving things tt-metal doesn't expose
(L2CPU boot, ARC PLL changes, raw NOC writes to L2CPU DRAM).

The two relevant entrypoints:

- `tt::umd::TTDevice::create(0)` — opens `/dev/tenstorrent/0` and gives
  back a handle. Cheap; doesn't initialize ARC telemetry. It's the
  bare-minimum "let me poke registers" object.
- `tt::umd::WarmReset::warm_reset_chip_id({0})` — issues a PCIe
  Secondary-Bus-Reset to the card. Equivalent to `tt-smi -r 0`. Required
  before some bring-up sequences.

The interesting methods on `TTDevice`:

- `write_to_device(buf, tt_xy_pair{x, y}, addr, size)` — NOC inbound
  write to tile (x, y).
- `read_from_device(...)` — NOC inbound read.
- `write_to_arc_apb(buf, arc_offset, size)` — write to a register on the
  ARC tile via a special path that doesn't go through the NOC. ARC
  registers control PLLs, resets, and similar chip-level state.

The PoC wraps these in `poc::common`'s `NocWrite32`, `BootL2cpu0`,
`SetL2cpuPll`, etc. — there's no magic, those are just thin helpers.

A subtle thing the PoC discovered: UMD's `write_to_arc_apb` takes
*offsets within the ARC APB peripheral region*, not the full AXI
address. `pyluwen.axi_write32(0x80030014)` and
`tt_device->write_to_arc_apb(..., 0x30014, ...)` target the same
register; the `0x80000000` AXI base is implicit in UMD. Get that wrong
and your "writes" go into dead BAR space and silently do nothing. Lots
of mysterious behaviour fell out of this single bug.

---

## 8. TTNN's tensor model in 90 seconds

You probably already know: TTNN is the high-level tensor library;
`ttnn::add(a, b)` runs an eltwise add on Tensix and gives you a `Tensor`.

What's new for the PoC is *where the bytes physically live*:

- A `Tensor` is backed by a `MeshBuffer`, which is backed (in the
  unit-mesh case) by a single-device `Buffer`.
- A DRAM-resident `Buffer` is **interleaved across DRAM banks**: pages
  are striped one-per-bank in round-robin. With 7 active banks and 16
  pages, page 0 lives in bank 0, page 1 in bank 1, ..., page 6 in bank 6,
  page 7 in bank 0 (second slot), and so on.
- The buffer has a single `address()` (a "DeviceAddr" — the offset
  inside any one bank where its pages start) and a `page_size()`. The
  per-bank stride between consecutive same-bank pages is
  `aligned_page_size = round_up(page_size, alignment)`.
- For `TILE` layout (which Tensix prefers), one page is one 32×32 tile.
  For FLOAT32, that's 4 KiB.

To compute a specific tensor element's physical address, you need:

```
page_idx     = element_byte_offset / page_size
elem_in_page = element_byte_offset % page_size
bank_id      = page_idx % num_banks
page_in_bank = page_idx / num_banks
addr_in_bank = bank_base[bank_id] + page_in_bank * aligned_page_size
                                  + elem_in_page
```

Step 4's firmware does exactly this loop, walking pages instead of
elements (since "increment every float in this page" doesn't depend on
which element-within-tile you're looking at).

The PoC needs one more thing TTNN doesn't put in a single API call:
**which NOC core does each bank live at?** That's `mesh_device->
logical_core_from_dram_channel(bank_id)` followed by `mesh_device->
virtual_core_from_logical_core(logical, tt::CoreType::DRAM)`. There are
several other-looking APIs that *almost* work (`virtual_noc0_coordinate`,
`allocator->get_logical_core_from_bank_id`) and silently return Tensix
coords for DRAM bank IDs — see the README quirks entry.

---

## 9. Synchronization between host and X280

The X280 firmware is a tiny "coprocessor" you talk to over a shared
control block in L2CPU0 DRAM. The pattern is always:

1. Host writes the request body (`data_addr`, `num_elems`, etc.) at some
   agreed offset.
2. Host writes a separate **kick** word (the "go" flag) at a different
   offset, *as a separate NOC transaction*. The split is needed because
   NOC packetisation can deliver the bytes of a single multi-byte write
   in offset order — if `kick` were at offset 0 of one big write, the
   X280 could see `kick = 1` *before* the trailing bytes of the same
   write land in DRAM.
3. X280 polls `kick` with acquire semantics, reads the body, does the
   work.
4. X280 writes a **done** word (a magic value, not just a flag) at a
   *different* offset from `kick`.
5. Host polls `done` until it sees the magic.

Why the offsets must differ: this silicon has an empirical quirk where
host NOC reads of any address the host *previously wrote* return the
host's last value forever, even after the X280 overwrites the same
address. Writes by the X280 to addresses the host has never touched are
visible normally. So `kick` (host-written) and `done` (X280-written)
have to be at different cache-line offsets — otherwise the host polls
`done` and only ever sees its own kick value.

The PoC's `Task` struct in `host/common.hpp` is laid out with this in
mind: `kick` at offset 0, `done` at offset 24. Step 4 has its own
`Step4Task` with the same property.

`__atomic_store_n(..., __ATOMIC_RELEASE)` on the X280 side is what
synchronises: prior stores in the firmware happen-before the `done`
write, so by the time the host sees `done == magic`, the data the
firmware was writing is also visible.

---

## 10. The boot dance, demystified

Step 1's `BootL2cpu0()` does eight things in a strict order:

1. **Warm reset the card.** Brings ARC and the L2CPU into a known state.
2. **Open a `TTDevice`.** No `init_tt_device()` — that one starts an
   ArcMessenger and would block waiting for telemetry that isn't ready
   yet on a fresh reset.
3. **Pre-boot NOC read of L2CPU0 DRAM.** If it returns `0xffffffff`, the
   GDDR slice is harvested and there's no point continuing — bail.
4. **Enable the L3 cache** (NOC write `0xf` to register `0x02010008`
   inside the L2CPU tile). Without this, the X280's instruction fetches
   stall on first use.
5. **Load firmware** (NOC write the `.bin` to `0x4000_3000_0000`).
6. **Program reset vectors** for all four harts to point at
   `0x4000_3000_0000` (NOC writes to a register at `0xfffff7fefff10000`
   inside the L2CPU tile, four core × two uint32 each).
7. **PLL down → deassert reset → PLL up.** ARC writes step PLL4 from
   whatever it was to 200 MHz, write the bit that releases the X280
   from reset, then step the PLL up to 1750 MHz. Stepping happens one
   feedback-divider unit per microsecond — faster transitions can
   wedge the AXI fabric.
8. **Configure L2 prefetchers** (four NOC writes per hart). Improves
   I/O bandwidth between L3 and DRAM but isn't strictly load-bearing.

Everything from step 4 onward must happen *after* the warm reset and
*before* the X280 starts running, because programming the reset vectors
on a running CPU does nothing useful.

---

## 11. The PoC steps mapped to the concepts above

- **step1_boot** uses §6 (boot a stub firmware), §7 (UMD raw access),
  §10 (boot dance). The X280 writes a magic uint32 to local DRAM via
  AXI; the host reads it via NOC inbound to (8, 3).

- **step2_task** adds §9 (kick/done synchronisation). Host writes a
  16-element float buffer to L2CPU0 DRAM, X280 reads via AXI, increments,
  writes back, signals done.

- **step3_ttnn_x280** adds §8 (TTNN tensors) and the trick of opening
  a *parallel* `TTDevice` while TTNN's `MeshDevice` is also alive — UMD
  allows it because the two consumers don't share NOC tiles or ARC state.
  TTNN computes 5.0 in Tensix DRAM, host `to_vector()`s it, stages it
  to L2CPU0 DRAM, X280 increments, host reads back from L2CPU0 DRAM.
  This is the "host-staging" version of the architecture, useful but
  not the goal.

- **step4_inplace** finally does §3, §4, §5 for real: it tells the X280
  the bank table for the TTNN tensor, the X280 programs one 128 GiB NOC
  TLB window per active bank, walks every page through that window
  with NOC reads/writes, increments in place. The host *never sees the
  tensor data during the modify phase* — TTNN's `to_vector()` only
  reads it after the X280 is done. This is the actual architectural
  win: the modify path is X280 → NOC → Tensix DRAM, no PCIe.

---

## 12. Silicon quirks worth keeping in mind

Each of these cost real time during the PoC bring-up; they're documented
in the README and in `MEMORY.md` (Claude's notes for future sessions),
but here's the short list:

1. **`mstatus.FS = 0` at boot** — set it in `_start` or any FP
   instruction dead-loops the firmware.
2. **Publication race** — `kick` at offset 0 + multi-byte body in one
   NocWrite is a race; publish `kick` separately.
3. **Host-write/read shadow** — host reads of any address it previously
   wrote return the host's value forever; use distinct offsets for
   `kick` and `done`.
4. **`cbo.flush` hangs the X280** — Zicbom CMOs are accepted but never
   complete on this silicon. Don't reach for them; rely on `__atomic_*`
   release-stores and Memory-Port/System-Port choice instead.
5. **`init_tt_device()` blocks on a fresh `ResetCard`** — ARC telemetry
   isn't ready. Skip it for the bring-up path; tt-metal-style code
   running after the chip is up doesn't have this problem.
6. **`virtual_noc0_coordinate(...)` is Tensix-flavoured** — for DRAM
   banks, it returns nonsense; use `virtual_core_from_logical_core(
   logical, tt::CoreType::DRAM)`.
7. **UMD `write_to_arc_apb` offsets are APB-relative** — strip
   `0x80000000` from any address you copy out of pyluwen-style code.
8. **TTNN runtime asset path** — set `TT_METAL_RUNTIME_ROOT`
   (not `TT_METAL_HOME`) before any TTNN call.

---

## Where to read more

- `tt-bh-linux/docs/addressing.md` — the canonical NOC TLB encoding doc.
- `tt-bh-linux/README.md` — the L2CPU memory map (System Port / Memory
  Port table).
- `tt-bh-linux/boot.py` — the reference implementation of the L2CPU
  boot sequence, in pyluwen. The PoC's `BootL2cpu0` is a UMD port of
  this.
- `tt-metal/.../umd/device/api/umd/device/arch/blackhole_implementation.hpp`
  — `DRAM_CORES_NOC0`, `TENSIX_CORES_NOC0`, ARC offsets, harvesting
  conventions. The source of truth for tile coordinates.
- `tt-metal/tt_metal/impl/buffers/buffer.cpp::page_address` — the exact
  formula for "where does page P of an interleaved buffer live in bank
  B." Step 4's firmware does the inverse of this.
