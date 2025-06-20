# TT-Alchemist Python CLI

A Python CLI wrapper for the tt-alchemist library.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Convert MLIR model to C++
tt-alchemist convert --input model.mlir --output-dir output/
```

## Environment Variables

- `TT_ALCHEMIST_LIB_PATH`: Path to the tt-alchemist shared library
