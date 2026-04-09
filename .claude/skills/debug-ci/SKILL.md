---
name: debug-ci
description: >
    Debug a failing GitHub Actions CI pipeline. Use when the user
    pastes a GitHub Actions URL, asks about CI failures, or wants
    to reproduce a failing test locally.
---

# Prerequisite Knowledge:
You must read everything CLAUDE.md and AGENTS.md before proceeding.
- ALL tests must be run after sourcing env: `source env/activate`
- `SYSTEM_DESC_PATH` env var needs to exist, if not: `export SYSTEM_DESC_PATH=ttrt-artifacts/system_desc.ttsys
- if `ttrt-artifacts/system_desc.ttsys` doesn't exit, run: `ttrt query --save-artifacts`
- pytests should always be run with `-sv` flags
- always build after making changes before running tests: `cmake --build build`

## Step 1: Get failing test(s) from pipeline

Get the failing test(s) using `gh *` commands. If there are multiple failing pipelines, collect failing test(s) for the hardware that's available.

Pipeline names will be in the form of `run-tests (<arch>, <build-type>, <# of pipeline>)`, eg: `run tests (<n150, tracy, 4>)`. We only care about `<arch>`.

To check we are on the right machine to debug, install and run `tt-smi` and grep for the `<arch>` name. If you can't find it, error out of the prompt.
```
source env/activate
pip install tt-smi
tt-smi | grep ...
```

## Step 2: Identify Test Types

The following test types are debuggable:
- ttnn-jit tests: any test found under `test/ttnn-jit`
- builder tests: any test found under `test/python`
- lit tests: any test found under `test/ttmlir` or with `FileCheck` in the test file

If the test was added by a commit in the current branch:
- understand all the changes made in current branch and why we are making them
- debug to your best effort

If this is a pre-existing test, then we broke something on main. First run the failing tests locally to gather identify the error. For any runtime, compilation, hangs, PCC errors, continue to Step 3. Else, debug to your best effort.

## Step 3: Compare IR to main
Most failures can be found by comparing IR differences between branch and main. First we need to find what the input IR was. Lit tests are just the test itself.

For ttnn-jit tests, running the pytest with `-sv` flag will dump the input IR
