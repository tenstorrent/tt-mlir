# PoC: X280 Big RISC-V Execution on Blackhole

## Context

CPU-hoisted ops in tt-mlir pay two PCIe roundtrips per op (device→host→device). The Blackhole chip has 4 L2CPU blocks with SiFive X280 RISC-V cores that share the NOC with Tensix. This PoC proves that the X280 can execute bare-metal code and modify tensor data, validating the architecture for eliminating PCIe overhead.

## Goal

A single C++ program that:
1. Uses TTNN to create tensors and perform `ttnn::add` on device
2. Stages the result to L2CPU DRAM via UMD
3. Boots an X280 bare-metal firmware that increments each element
4. Reads back and prints the modified data

## Prerequisites

- Machine with Blackhole P100/P150 card, `tt-kmd` loaded
- tt-metal built (provides TTNN, Metalium, UMD libraries)
- `riscv64-linux-gnu-gcc` (Ubuntu: `apt install gcc-riscv64-linux-gnu`)
  - The `-linux-gnu` toolchain works for bare-metal with `-nostdlib -nostartfiles -static`

## Plan: 4 Incremental Steps

Each step compiles, runs on hardware, and is verified before moving to the next.

---

### Step 1: Boot X280, write/read a magic value

**What:** Minimal C++ program that uses UMD to write a value to L2CPU DRAM, boot the X280 with a trivial firmware, and read back a modified value.

**Files to create:**
- `poc/step1_boot.cpp` — host program
- `poc/fw_step1.c` — firmware (writes 0xDEADBEEF to a known address)
- `poc/fw_start.S` — startup assembly (set stack, park non-zero harts)
- `poc/fw.ld` — linker script
- `poc/Makefile`

**Host program (`step1_boot.cpp`):**
```
- Open TTNN device (initializes MetalContext, gives us the Cluster singleton)
- Get Cluster via MetalContext::instance().get_cluster().get_driver()
- Get L2CPU core coordinates from SocDescriptor
- Write firmware binary to L2CPU DRAM at 0x4000'3000'0000
- Write 0x0 to a "mailbox" address (L2CPU DRAM + 1MB)
- Enable L3 cache (write 0xF to 0x0201'0008 via NOC)
- Program reset vectors (write entry address to 0xfffff7fefff10000)
- Trigger L2CPU reset (ARC APB write to offset 0x30014)
- Poll mailbox until value == 0xDEADBEEF
- Print success/failure
- Close device
```

**Firmware (`fw_step1.c`):**
```
void main(void) {
    volatile uint32_t *mailbox = (volatile uint32_t *)0x400030100000ULL;
    *mailbox = 0xDEADBEEF;
    while(1) asm volatile("wfi");
}
```

**Startup (`fw_start.S`):**
```
_start:
    csrr a0, mhartid
    bnez a0, park
    la sp, _stack_top
    call main
park:
    wfi
    j park
```

**Linker script (`fw.ld`):**
```
ENTRY(_start)
SECTIONS {
    . = 0x400030000000;
    .text : { *(.text.start) *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) }
    .bss : { *(.bss*) *(COMMON) }
    . = ALIGN(16);
    . = . + 0x4000;
    _stack_top = .;
}
```

**Build:** `riscv64-linux-gnu-gcc -march=rv64imafdc -mabi=lp64d -nostdlib -nostartfiles -static -T fw.ld -O2 -o fw.elf fw_start.S fw_step1.c && riscv64-linux-gnu-objcopy -O binary fw.elf fw.bin`

**Verify:** Program prints "X280 booted, mailbox = 0xDEADBEEF". If it times out, debug by reading L2CPU DRAM regions to see if firmware loaded correctly.

**Key files to reference:**
- `tt-bh-linux/boot.py:70-79` — reset sequence
- `tt-bh-linux/boot.py:219-230` — reset vector programming
- `tt-bh-linux/console/l2cpu.cpp:18-44` — NOC coordinates and DRAM addresses
- `tt-metal/ttnn/examples/add/add.cpp` — TTNN device open pattern
- `tt-metal/tt_metal/impl/context/metal_context.hpp` — MetalContext::instance()
- `tt-umd cluster.hpp` — Cluster::write_to_device(), read_from_device()

---

### Step 2: Task dispatch — X280 modifies an array in its own DRAM

**What:** Extend Step 1. Host writes a float array + task descriptor to L2CPU DRAM. Firmware reads task, increments each element, signals done. Host reads back.

**Files to modify/create:**
- `poc/step2_task.cpp` — host program
- `poc/fw_step2.c` — firmware with task loop

**Host program (`step2_task.cpp`):**
```
- Same device/cluster setup as Step 1
- Write 16 floats (all 5.0) to L2CPU DRAM + 2MB ("data area")
- Write task descriptor to L2CPU DRAM + 1MB:
    { status=READY, data_addr, num_elems=16 }
- Load and boot firmware (same as Step 1)
- Poll task.status until DONE
- Read 16 floats back from data area
- Print (expect all 6.0)
```

**Task descriptor struct (shared between host and firmware):**
```c
struct Task {
    volatile uint32_t status;  // 0=idle, 1=ready, 2=running, 3=done
    uint32_t pad;
    uint64_t data_addr;
    uint64_t num_elems;
};
```

**Firmware (`fw_step2.c`):**
```
void main(void) {
    volatile struct Task *task = (void*)TASK_ADDR;
    while (__atomic_load_n(&task->status, __ATOMIC_ACQUIRE) != 1)
        asm volatile("wfi");
    task->status = 2;
    float *data = (float*)task->data_addr;
    for (uint64_t i = 0; i < task->num_elems; i++)
        data[i] += 1.0f;
    __atomic_store_n(&task->status, 3, __ATOMIC_RELEASE);
    while(1) asm volatile("wfi");
}
```

**Verify:** Prints 16 values, all 6.0.

---

### Step 3: TTNN + X280 — full PoC

**What:** TTNN creates tensors, performs add on Tensix, result is staged to L2CPU DRAM, X280 increments, host reads back.

**Files to modify/create:**
- `poc/step3_ttnn.cpp` — full PoC host program (builds on step2)
- Reuse `fw_step2.c` firmware unchanged

**Host program (`step3_ttnn.cpp`):**
```
- Open TTNN device
- a = ttnn::full({1,1,4,4}, 2.0f, FLOAT32, ROW_MAJOR, device, DRAM_MEMORY_CONFIG)
- b = ttnn::full({1,1,4,4}, 3.0f, FLOAT32, ROW_MAJOR, device, DRAM_MEMORY_CONFIG)
- result = ttnn::add(a, b)   // all 5.0, lives in Tensix DRAM
- host_data = result.cpu(true).to_vector<float>()  // read to host
- Print "Before X280:" + first 8 values (expect 5.0)
- Get Cluster, boot X280 (same as Step 2)
- Write host_data to L2CPU DRAM data area via cluster.write_to_device()
- Write task descriptor, poll for completion
- Read modified data back from L2CPU DRAM
- Print "After X280:" + first 8 values (expect 6.0)
- Close device
```

**Why stage through host:** Interleaved DRAM buffers stripe pages across 8 banks. Direct NOC cross-read requires iterating all banks or using SINGLE_BANK allocation. Staging via host is simple and correct for PoC. The data transfer is:
- Tensix DRAM → host (PCIe, same as current CpuOp)
- host → L2CPU DRAM (PCIe via UMD write_to_device)
- X280 modifies in L2CPU DRAM (zero PCIe)
- L2CPU DRAM → host (PCIe via UMD read_from_device)

This is NOT the final architecture (which eliminates PCIe entirely), but it proves every component works together.

**Verify:** Prints before=5.0 and after=6.0 for all elements.

---

### Step 4 (stretch): X280 reads Tensix DRAM directly via NOC

**What:** Instead of staging through host, configure X280's own NOC TLB window to read from a Tensix DRAM bank and modify in-place.

**Prerequisite knowledge from step 3:** tensor's DRAM bank NOC coordinates and buffer address (from `buffer->address()`, `device->logical_core_from_dram_channel()`).

**This step requires:**
- Understanding page interleaving across banks (or using a 1-page tensor)
- Getting the X280 NOC TLB register encoding right (documented in `docs/addressing.md`)
- Handling the fact that `riscv64-linux-gnu-gcc` might not support `__attribute__((packed))` bitfield structs identically to the hardware layout

**Approach:** Use a tiny tensor (e.g., 16 floats = 64 bytes, single page) so it lands in one DRAM bank. Pass the bank's NOC (x,y) and the buffer offset to the firmware via the task descriptor.

**Firmware changes:**
```c
// Configure 2MB TLB window 0 to point at Tensix DRAM bank
volatile uint32_t *tlb_cfg = (volatile uint32_t *)0x2ff00000;
// Encode: addr[42:21]=0 (bank starts at 0), x_end=noc_x, y_end=noc_y
uint64_t reg_lo = (uint64_t)0;  // addr bits [42:0] = 0 (aligned to 2MB)
uint64_t reg_hi = ((uint64_t)noc_x << 0) | ((uint64_t)noc_y << 6);  // x_end, y_end
tlb_cfg[0] = (uint32_t)(reg_lo);
tlb_cfg[1] = (uint32_t)(reg_lo >> 32);
tlb_cfg[2] = (uint32_t)(reg_hi);

// Access tensor through the window (System Port, uncached)
float *tensor = (float *)(0x30400000000ULL + buffer_offset);
for (uint64_t i = 0; i < num_elems; i++)
    tensor[i] += 1.0f;
```

**Host changes:**
- Get tensor address: `mesh_buffer->address()`
- Get DRAM bank core: `device->logical_core_from_dram_channel(0)` → NOC coords
- Pass both to X280 via task descriptor
- After X280 modifies in-place, read tensor back via TTNN `result.cpu()` (goes through normal Tensix DRAM read path)

**Verify:** Same expected output as Step 3, but without host staging. The tensor data round-trips: TTNN writes to Tensix DRAM → X280 modifies via NOC → TTNN reads back.

---

## File Structure

```
poc/
  Makefile              # builds firmware + host program
  fw_start.S            # shared startup assembly
  fw.ld                 # shared linker script
  fw_step1.c            # Step 1 firmware
  fw_step2.c            # Step 2-3 firmware
  fw_step4.c            # Step 4 firmware (NOC TLB)
  step1_boot.cpp        # Step 1 host
  step2_task.cpp        # Step 2 host
  step3_ttnn.cpp        # Step 3 host (full PoC)
  step4_noc.cpp         # Step 4 host (stretch)
  CMakeLists.txt        # for host programs (links TTNN + UMD)
```

## Key Constants (shared header)

```c
// L2CPU0 NOC coordinates: (8, 3) — from tt-umd blackhole_implementation.hpp
// L2CPU0 DRAM base: 0x4000'3000'0000 — from tt-bh-linux/console/l2cpu.cpp
// Reset vector base: 0xfffff7fefff10000 — from tt-bh-linux/boot.py
// L3 cache enable reg: 0x0201'0008 — from tt-bh-linux/boot.py
// ARC APB L2CPU reset offset: 0x30014 — from tt-bh-linux/boot.py
// CLINT MSIP: 0x0200'0000 — standard RISC-V
// X280 2MB TLB config base: 0x2ff00000 — from tt-bh-linux/docs/addressing.md
// X280 2MB window base (System Port): 0x30400000000 — from docs/addressing.md
```

## Build & Run

```bash
# Install RISC-V toolchain (if not present)
sudo apt install gcc-riscv64-linux-gnu

# Build firmware
cd poc && make firmware

# Build host program (needs tt-metal build environment)
cmake -B build -DCMAKE_PREFIX_PATH=$TT_METAL_HOME/build/lib/cmake
cmake --build build

# Run each step
./build/step1_boot
./build/step2_task
./build/step3_ttnn
```

## Risk Log

| Risk | Impact | Mitigation |
|------|--------|------------|
| L2CPU harvested on test machine | Blocker | Check `soc.get_cores(CoreType::L2CPU)` first, fail gracefully |
| PLL not at safe frequency before reset | X280 hangs | Skip PLL changes initially — ARC firmware may set a default. Add PLL control if needed |
| `riscv64-linux-gnu-gcc` can't link bare-metal | Build fails | Use `-nostdlib -nostartfiles -static -T fw.ld`. If issues, install `riscv64-unknown-elf-gcc` |
| NOC TLB encoding wrong (Step 4) | X280 hangs/reads garbage | Debug incrementally: first read a known DRAM address that's been initialized, compare expected vs actual |
| TTNN holds device lock that blocks UMD writes | UMD write hangs | Separate fd to `/dev/tenstorrent/0`; or ensure TTNN ops complete before UMD access |
| tt-metal Cluster wrapper differs from UMD Cluster API | Compile error | The tt-metal `Cluster` class wraps UMD; use `get_driver()` to get the underlying `tt::umd::Cluster` |
