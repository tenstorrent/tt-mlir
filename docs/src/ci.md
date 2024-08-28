# `ci`

Our CI infrastructure is currently hosted on cloud. Cloud machines are used and linked as GitHub runners.

## Key Words
### Target Silicon (coming soon)
```bash
- 1:1 mapping to unique system-desc (this is because an n150 card can have different harvested rows)
```

### Target Family
```bash
- product type (n150, n300)
```

### Target Capabilities (coming soon)
```bash
- describes testable traits of Target Family
n150: {
    test params to use if running on n150
}
n300: {
    test params to use if running on n150
}
```

### Test Capabilities (coming soon)
```bash
- set of target capabilities defined in the test
- test will populate certain parameters depending on the Target Family/Target Silicon it is running on
```

## GitHub Runner CI Tags
### Runner Use
There are 2 types of runner machines. Builders build offline and runners are silicon machines.
```bash
- builder
- runner
```

### Runner Type
There are 2 runner types. Bare metals are standalone and virtual machines are kubernetes pods.
```bash
- bare-metal
- virtual-machine
```

### Architecture
Supported architectures
```bash
- wormhole_b0
- blackhole (coming soon)
```

### Pipeline Type
Supported pipelines
```bash
- perf
- functional
```

### Active
Defines whether a runner is in service or taken out of service for maintenance
```bash
- in-service
- out-of-service
```

### Target Family
Supported configurations of machines
```bash
- n150
- n300
- t3000 (coming soon)
- tg (coming soon)
- tgg (coming soon)
```

### Target Silicon (coming soon)
```bash
-silicon-n150-0 (0th row harvested)
-silicon-n150-1 (1th row harvested)
-silicon-n300-0-0 (0th row harvested both chips)
```

## Pipeline durations
```bash
- push: every push to main
- pr: every PR
```

## CI Test Flow
```bash
1. GitHub runner
- build tt-mlir
- build ttrt
- upload artifacts

2. Silicon runner
- download tt-mlir / ttrt artifacts
- ttrt generate system desc
- llvm-lit runs all unit test, including silicon ones to generate flatbuffers (will only generate ones that are supported for that test file)
- ttrt runs generated flatbuffers
```

## Adding a test
When adding a test, you can specify when the test should run and what values it should inherit. The test defines how it should run, not the infrastructure. The infrastructure will execute what the test defines. For now, if you specify nothing, it will run on all default parameters.
Note: if you provide a target family, then it will be default run on any target silicon machine. If you need a specific target silicon machine (eg one with 1st row harvested), specify it in Target Silicon.
Note: if you specify perf pipeline, it will automatically run on a bare metal machine
Default parameters
```bash
[Architecture]: [wormhole_b0]
[Pipeline]: [functional, perf]
[Target Family]: [n150, n300]
[Target Silicon]: []
[Duration]: [push]
```

```bash
Location: test/ttmlir/Silicon
File Type: .mlir
REQUIRES: [Architecture] [Pipeline] [Target Family] [Target Silicon] [Duration] (coming soon)
UNSUPPORTED: [Target Family] [Target Silicon] (coming soon)
```
