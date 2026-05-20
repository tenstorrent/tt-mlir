# X280 on Blackhole PoC

Proves the SiFive X280 RISC-V core inside L2CPU0 can boot bare-metal
firmware, modify tensor data, and round-trip with both TTNN and the host —
in the same process, without resetting the chip between. Validates the
architecture for eliminating PCIe round-trips on CPU-hoisted ops in
tt-mlir. See `plan.md` for the original design notes.

## Layout

```
poc/
  fw/                       # bare-metal X280 firmware
    fw.ld                   # linker script (code @ 0x400030000000)
    fw_start.S              # startup: park non-zero harts, enable FP, jump to fw_main
    fw_step1.c              # Step 1: write 0xDEADBEEF to a mailbox
    fw_step2.c              # Step 2 / 3: increment a host-supplied float buffer
    fw_step4.c              # Step 4: NOC-TLB program + in-place increment of an interleaved tensor
    fw_step5.c              # Step 5: copy in/out + call MLIR-lowered kernel from tmp/cpu.o
    fw_step6.c              # Step 6: persistent task loop, dispatches arbitrary kernels from a code blob
    cpu2.ll                 # LLVM IR for abs + matmul kernels with helpers (from CPU-hoisting pipeline)
    cpu2_dispatch.c         # Dispatch entry point + memcpy for the code blob
    cpu2_blob.ld            # Linker script for the code blob (linked at CODE_LOAD_ADDR)
    Makefile                # cross-build via clang-20 + ld.lld + llvm-objcopy
  host/
    common.hpp common.cpp   # shared boot helpers (PLL, reset, NOC, ARC APB)
    step1_boot.cpp          # Step 1 host
    step2_task.cpp          # Step 2 host
    step3_ttnn_x280.cpp     # Step 3: TTNN -> host -> X280 -> host (host staging)
    step4_inplace.cpp       # Step 4: TTNN -> X280 reaches Tensix DRAM via NOC TLB -> TTNN (no host staging)
    step5_hoisted.cpp       # Step 5: TTNN -> X280 runs MLIR-lowered abs (linked from tmp/cpu.o) -> TTNN
    step6_multi_kernel.cpp  # Step 6: dispatches abs + matmul through generic firmware loop
  CMakeLists.txt
```

## Steps

| Binary             | What it proves |
|--------------------|----------------|
| `step1_boot`       | X280 boots, executes a 4-instruction `fw_main`, host sees the mailbox flip |
| `step2_task`       | Host publishes a Task descriptor; X280 reads a 16-float buffer, increments each, signals done |
| `step3_ttnn_x280`  | TTNN computes 5.0, host stages to L2CPU0 DRAM, X280 increments to 6.0, host reads back |
| `step4_inplace`    | TTNN computes 5.0; X280 reaches into the **interleaved Tensix DRAM** banks directly via its own NOC TLB and increments to 6.0 in place; TTNN reads back. No PCIe round-trip during the modify phase. |
| `step5_hoisted`    | TTNN allocates `input(-5.0)` and `output(0.0)` tensors; X280 stages input into local DRAM, calls **MLIR-compiled `abs` kernel from `poc/fw/cpu.o`** linked into the firmware, writes results back to the output tensor. TTNN reads `5.0`. First step where the X280 executes compiler-generated code. |
| `step6_multi_kernel` | Generic multi-kernel dispatch. Firmware runs a **persistent task loop** with sequence-number handshake. Kernel code (abs + matmul with helpers) is compiled into a **separate code blob** loaded at runtime, not linked into the firmware. Host dispatches two tasks sequentially: abs(-5.0)→5.0 then matmul(3.0×2.0×4000)→24000.0, exercising different argument counts and tensor shapes. |

## Prereqs

- `clang-20` + `ld.lld` + `llvm-objcopy-20` + `llc-20` on PATH for the firmware cross-build
  (any toolchain that supports `--target=riscv64-unknown-elf` works; see
  `fw/Makefile` for how to point it at a different toolchain).
- tt-metal built at `<tt-mlir>/third_party/tt-metal/src/tt-metal/build`,
  *and* the tt-alchemist standalone install populated at
  `<tt-mlir>/build/tools/tt-alchemist/templates/cpp/standalone/ttnn-install/`
  (only the latter is needed for the TTNN-linked steps; step1 / step2 only
  need the former).
- tt-kmd loaded, card visible at `/dev/tenstorrent/0`.
- L2CPU0 not harvested. step1 / step2 / step3 fail loudly with
  `pre-boot NOC read returned 0xffffffff` if it is.

## Build

```
cd poc
CC=clang-20 CXX=clang++-20 cmake -B build -G Ninja
cmake --build build
```

Clang for the host build matches what tt-metal uses. With g++ the C++20
concept-mangling of `requires` clauses differs and links against TTNN fail
with `undefined reference to ttnn::full<float>(...)`.

If TTNN/Metalium aren't found, only step1 / step2 build (CMake prints a
status line and skips the step3 targets).

## Run

```
build/bin/step1_boot       build/fw/fw_step1.bin
build/bin/step2_task       build/fw/fw_step2.bin
build/bin/step3_ttnn_x280  build/fw/fw_step2.bin
build/bin/step4_inplace    build/fw/fw_step4.bin
build/bin/step5_hoisted    build/fw/fw_step5.bin
build/bin/step6_multi_kernel build/fw/fw_step6.bin build/fw/cpu2_blob.bin
```

Each binary issues a warm reset on entry and on exit, so consecutive runs
are independent. If something hangs, `tt-smi -r 0` and retry.

The TTNN-linked binaries seed `TT_METAL_RUNTIME_ROOT` at startup so you
don't need to source `env/activate` first.

## Silicon quirks worth knowing

A handful of non-obvious things found while bringing this up. Anyone
extending the PoC (step 4, real CPU hoisting integration) should read
these before debugging:

- **`mstatus.FS = 0` at boot** — the X280 boots with the FP unit off; any
  `flw` / `fadd.s` / `fmv.*` traps to `mtvec`, which is unset, dead-looping
  the firmware. `fw_start.S` sets `mstatus.FS = Dirty` before calling
  `fw_main`.

- **Task publication race** — putting `kick` at offset 0 and writing the
  whole Task in one NocWrite lets the X280 acquire `kick = kReady` before
  trailing fields land in DRAM. The host writes the body, then publishes
  `kick` as a separate `NocWrite32`.

- **Host write/read shadow** — once the host writes a DRAM word via NOC,
  subsequent host reads of that word return the host's last value forever,
  even after the X280 overwrites it. Other offsets in the same cache line
  behave normally. The Task struct puts `kick` (host→fw) and `done`
  (fw→host) at distinct offsets for this reason; the host *must never*
  write to `done`.

- **`cbo.flush` hangs** — the Zicbom CMO is accepted (no illegal-instr
  trap) but doesn't complete on this silicon. Don't use it. Memory Port
  reads are sufficient for fw→host visibility on addresses the host
  hasn't tainted (see shadow rule above).

- **`init_tt_device()` after a fresh `ResetCard`** — blocks waiting for
  ARC telemetry that's not ready. Step1 and step2 use a pre-boot NOC read
  of L2CPU0 DRAM (`0xffffffff` ⇒ harvested) as the harvest sanity check
  instead of telemetry.

- **UMD `write_to_arc_apb` takes APB-region offsets, not full AXI
  addresses** — `pyluwen.axi_write32(0x80030014)` is the same register as
  `tt_device->write_to_arc_apb(..., 0x30014, ...)`. Strip the
  `0x80000000` AXI base when porting code from `tt-bh-linux/boot.py`.

- **Two `TTDevice::create(0)` calls coexist** — step3 keeps TTNN's
  MeshDevice open and opens a parallel UMD `TTDevice` to drive L2CPU0
  directly. UMD allows concurrent opens on the same chardev as long as
  the two consumers don't share NOC tiles or ARC state (Tensix vs L2CPU0
  here).

- **TTNN runtime asset path** — TTNN/Metalium reads `TT_METAL_RUNTIME_ROOT`
  (not `TT_METAL_HOME` — that's a legacy alias) to find SoC YAMLs.
  CMake bakes the absolute path in via a compile definition.

- **DRAM channel → NOC core: don't use `virtual_noc0_coordinate`.** That
  one is Tensix-flavoured and silently returns nonsense for DRAM logical
  coords. `allocator->get_logical_core_from_bank_id(bank_id)` is also
  wrong for DRAM — returns a Tensix worker logical, which then fails the
  bounds check in `virtual_core_from_logical_core`. The right path
  (used by step4) is `mesh_device->logical_core_from_dram_channel(bank_id)`
  → `mesh_device->virtual_core_from_logical_core(logical, tt::CoreType::DRAM)`.

- **X280 NOC TLB: System Port = uncached, Memory Port = cached.** Step 4
  uses the System Port window base (`0x80430000000 + N * 128GB`) so X280
  stores go straight out as NOC packets without sitting in L3.
