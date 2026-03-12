[![Tests][tests badge]][tests]
[![Codecov][codecov badge]][codecov]
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-mlir)

<div align="center">

<h1>

[Hardware](https://tenstorrent.com/cards/) | [Documentation](https://docs.tenstorrent.com/tt-mlir/) | [Discord](https://discord.gg/tenstorrent) | [Join Us](https://boards.greenhouse.io/tenstorrent?gh_src=22e462047us) | [Bounty $](https://github.com/tenstorrent/tt-mlir/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty)

</h1>

<img src="./docs/public/images/tt-mlir-logo.png" alt="ttmlir logo" height="230"/>

<br>

tt-mlir is a compiler framework that defines MLIR dialects for compiling and optimizing AI workloads on Tenstorrent accelerators.
Built on [LLVM's MLIR](https://mlir.llvm.org/), it transforms models from frontends like [TT-XLA](https://github.com/tenstorrent/tt-xla) and [TT-Forge-ONNX](https://github.com/tenstorrent/tt-forge-onnx) into optimized binaries targeting [TT-Metalium](https://github.com/tenstorrent/tt-metal).

> **Part of the [TT-Forge](https://github.com/tenstorrent/tt-forge) AI compiler ecosystem.**

</div>
<br>

-----
# Quick Links
- [Getting Started](https://docs.tenstorrent.com/tt-mlir/getting-started.html)
- [Tools](https://docs.tenstorrent.com/tt-mlir/tools.html)
- [Additional Reading](https://docs.tenstorrent.com/tt-mlir/additional-reading.html)

-----
# What is this Repo?

tt-mlir is an open-source compiler framework that is used to compile and optimize machine learning models for Tenstorrent's custom AI hardware, such as Wormhole and Blackhole. Built on top of LLVM’s MLIR, it defines custom dialects and transformation passes tailored for Tenstorrent's architectures. It enables efficient translation of models from front-end dialects, such as StableHLO, into binaries optimized for Tenstorrent accelerators.

-----
# Project Goals

- **Generality**: Support a wide range of AI models and workloads including training
- **Scalable**: First class primitives to describe scaling to multichip systems
- **Performant**: Enable great out of the box performance
- **Tooling**: Enable human in the loop guided compiler optimization
- **Open Source**: All project development is done in the open

[codecov]: https://codecov.io/gh/tenstorrent/tt-mlir
[tests]: https://github.com/tenstorrent/tt-mlir/actions/workflows/on-push.yml?query=branch%3Amain
[codecov badge]: https://codecov.io/gh/tenstorrent/tt-mlir/graph/badge.svg
[tests badge]: https://github.com/tenstorrent/tt-mlir/actions/workflows/on-push.yml/badge.svg?query=branch%3Amain

-----
# Related Tenstorrent Projects
- [TT-Forge](https://github.com/tenstorrent/tt-forge) — Central hub for the TT-Forge compiler project (demos, benchmarks, releases)
- [TT-XLA](https://github.com/tenstorrent/tt-xla) — Primary frontend for PyTorch and JAX (single and multi-chip)
- [TT-Forge-ONNX](https://github.com/tenstorrent/tt-forge-onnx) — Frontend for ONNX, TensorFlow, and PaddlePaddle (single-chip)
- [TT-Metalium](https://github.com/tenstorrent/tt-metal) — Low-level programming model and kernel development
- [TT-Blacksmith](https://github.com/tenstorrent/tt-blacksmith) — Optimized training recipes and experiments

-----
# Tenstorrent Bounty Program Terms and Conditions

This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-forge, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both “bounty” and difficulty level!
