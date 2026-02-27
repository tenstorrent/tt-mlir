# Running Virtualized Ubuntu VM on macOS

In some cases, like running a software simulated device, it can be beneficial
to run the stack on a local macOS machine.  This document covers the necessary
setup and configuration steps to get a performant Ubuntu VM setup on Apple
Silicon.

## Prerequisite Steps

1. [UTM](https://mac.getutm.app) is the VM application we'll be using in this
guide, so the first step is to download and install UTM.
2. [Ubuntu 22.04 ARM image download](https://cdimage.ubuntu.com/releases/22.04/release/).
  - Direct link: [64-bit ARM (ARMv8/AArch64) server install image](https://cdimage.ubuntu.com/releases/22.04/release/ubuntu-22.04.5-live-server-arm64.iso)

## UTM Setup

1. Launch UTM and click the `+` button to start a new VM.
2. Choose `Virtualize` (emulation works, but is unusably slow).
3. Under `Preconfigured` choose `Linux`.
4. Check box `Use Apple Virtualization` and select the ubuntu iso we just
   downloaded for `Boot ISO Image`.
  - Optionally check `Enable Rosetta` which can enable running ELF's compiled
    for x86 if you're interested.  It's not required and [additional steps](https://docs.getutm.app/advanced/rosetta/)
    are required for it to work.
5. This step depends on your machine's capabilities, but it's recommended to
   give 16GB of memory and to use the default CPU Cores setting. Note this can
   be changed after initial setup if you want to go back and tweak settings.
6. It's recommended to at least 128GB of storage, with LLVM installation and
   full SW stack we quickly reach 80 gigs of storage.
7. Optionally choose a shared host/VM directory.
8. Optionally name your new VM `ubuntu 22.04 arm64`

## VM Setup

1. Boot your newly created VM!
2. Run through the Ubuntu setup as you see fit, be sure that openssh is enabled
   which simplifies logging into your VM, but the rest of the defaults are
   sufficient.
3. If you plan on using your VM via ssh you can retrieve the ip address `ip a`
   and looking at the `inet` row under `enp0s1`.  Should look something like
   `inet 192.168.64.3`.  Another tip is to add this to the host's `~/.ssh/config`.
4. Install your normal developer tools as you see fit.

## Software Stack Installation

The majority of the software install flow is the same, with the exception of a
few caveats called out here.

1. Installing metal deps needs the additional flags below:
```bash
git clone git@github.com:tenstorrent/tt-metal.git
cd tt-metal
sudo bash install_dependencies.sh --docker --no-distributed
```
  - `--docker`: Despite not being in a docker, this is the flag that turns off
    configuring hugepages which is not required for VM.
  - `--no-distributed`: Currently the metal distributed feature requires a package
    version of openmpi that only supports x86.
2. Install tt-mlir system dependencies as outlined [by this step](./getting-started.md#ubuntu).
3. The environment needs to be built manually as outlined [here](./getting-started.md#setting-up-the-environment-manually).
4. We can then [build tt-mlir per usual](./getting-started.md#building-the-tt-mlir-project).
5. If planning to run tests on software sim, let's [build the ttrt tool](./ttrt.md#building).
  - The following all works per usual:
    - [Querying a system desc](./ttrt.md#generate-a-flatbuffer-file-from-compiler).
    - [Running a flatbuffer](./ttrt.md#run)
