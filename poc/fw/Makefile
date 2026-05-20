# Cross-build for X280 bare-metal firmware.
#
# Defaults to clang-20 + ld.lld + llvm-objcopy because that's what's on this
# host. To use the riscv64-linux-gnu GCC toolchain instead, build with:
#   make CC=riscv64-linux-gnu-gcc OBJCOPY=riscv64-linux-gnu-objcopy LDFLAGS="-T fw.ld -Wl,--build-id=none"
# (drop the `-fuse-ld=lld` and `--target=` from CFLAGS in that case).
#
# -nostdlib/-nostartfiles/-static keeps us free of any libc; -mcmodel=medany
# gives us the full 64-bit address range needed for the 0x400030000000 link
# address.

# `?=` doesn't help: make's implicit rules pre-set CC to `cc`. Command-line
# overrides (`make CC=riscv64-linux-gnu-gcc ...`) still beat `:=`.
CC      := clang-20
OBJCOPY := llvm-objcopy-20
OBJDUMP := llvm-objdump-20

CFLAGS  := --target=riscv64-unknown-elf \
           -march=rv64imafdc -mabi=lp64d -mcmodel=medany \
           -O2 -ffreestanding -nostdlib -nostartfiles -static \
           -Wall -Wextra -fno-common
LDFLAGS := -fuse-ld=lld -T fw.ld -Wl,--build-id=none

# Step 5 pulls in cpu.o, which was emitted by tt-mlir's CPU-hoisting pipeline
# with `-mcpu=sifive-x280` and therefore advertises the full X280 ISA string
# (rv64imafdcv + zba/zbb/zfh/...). Bumping the firmware build to the same
# CPU avoids ld.lld merge-attribute mismatches when linking the two together.
STEP5_CFLAGS := --target=riscv64-unknown-elf \
                -mcpu=sifive-x280 -mabi=lp64d -mcmodel=medany \
                -O2 -ffreestanding -nostdlib -nostartfiles -static \
                -Wall -Wextra -fno-common

# MLIR-lowered kernel object. Checked in here as a binary blob; regenerate
# from a matching ttir source via tt-mlir's CPU-hoisting pipeline + llc
# (`llc -mtriple=riscv64-unknown-elf -mcpu=sifive-x280 -target-abi=lp64d
# -filetype=obj cpu.ll -o cpu.o`) when the kernel changes.
CPU_O ?= cpu.o

# LLVM toolchain for compiling the kernel LLVM IR to a RISC-V object.
LLC     := llc-20

BINS := fw_step1.bin fw_step2.bin fw_step4.bin fw_step5.bin fw_step6.bin \
        cpu2_blob.bin

.PHONY: all clean
all: $(BINS)

fw_step1.elf: fw_start.S fw_step1.c fw.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ fw_start.S fw_step1.c

fw_step2.elf: fw_start.S fw_step2.c fw.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ fw_start.S fw_step2.c

fw_step4.elf: fw_start.S fw_step4.c fw.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ fw_start.S fw_step4.c

fw_step5.elf: fw_start.S fw_step5.c $(CPU_O) fw.ld
	$(CC) $(STEP5_CFLAGS) $(LDFLAGS) -o $@ fw_start.S fw_step5.c $(CPU_O)

fw_step6.elf: fw_start.S fw_step6.c fw.ld
	$(CC) $(STEP5_CFLAGS) $(LDFLAGS) -o $@ fw_start.S fw_step6.c

# Kernel code blob: compiled from cpu2.ll (LLVM IR from the compiler's CPU
# hoisting pipeline, includes kernels + helpers + x280_cpu_dispatch) plus a
# small runtime support file that provides memcpy for the freestanding
# environment. Linked at CODE_LOAD_ADDR so the firmware can call the
# dispatch entry point directly.
cpu2.o: cpu2.ll
	$(LLC) -mtriple=riscv64-unknown-elf -mcpu=sifive-x280 \
	       -target-abi=lp64d -filetype=obj $< -o $@

cpu2_rt.o: cpu2_rt.c
	$(CC) $(STEP5_CFLAGS) -c -o $@ $<

cpu2_blob.elf: cpu2.o cpu2_rt.o cpu2_blob.ld
	$(CC) $(STEP5_CFLAGS) -fuse-ld=lld -T cpu2_blob.ld -Wl,--build-id=none \
	      -o $@ cpu2.o cpu2_rt.o

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

%.lst: %.elf
	$(OBJDUMP) -d $< > $@

clean:
	$(RM) *.elf *.bin *.lst cpu2.o cpu2_rt.o
