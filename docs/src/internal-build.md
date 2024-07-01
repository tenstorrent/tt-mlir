# Internal Build Notes / IRD

- When building the runtime we must use Ubuntu 20.04 docker image
  - When making an IRD reservation use `--docker-image
    yyz-gitlab.local.tenstorrent.com:5005/tenstorrent/infra/ird-ubuntu-20-04-amd64:latest`
- You'll have to manaully install a newer version of cmake, at least 3.20, the easiest way to do this is to `pip install cmake` and make sure this one is in your path
- You'll want LLVM installation to persist IRD reservations, you can achieve this by:
  - mkdir /localdev/$USER/ttmlir-toolchain
  - When requesting an IRD use `--volumes /localdev/$USER/ttmlir-toolchain:/opt/ttmlir-toolchain`
