# `CI`

Our CI infrastructure is currently hosted on cloud. Cloud machines are used and linked as GitHub runners.

CI is triggered by new pull request and on push into main (usually when PR is merged).

CI is designed to automatically collect analytics data for each workflow run, including test results and code coverage. It will also publish newest release of documentation on GitHub.

## Builds
CI performs the following build jobs:

> - Release build "__speedy__" - release image optimized for speed.
> - Release build "__tracy__" - release image with runtime trace/debug capabilities including performance measurments.
> - Debug build with unit tests and test coverage.
> - CLang tidy
> - ...and [Tests](#testing)

The build of tt-mlir is done using build-tt-mlir-action.
Only the Debug build has a specific implementation because it is also used to run unit tests and collect and publish code coverage data.
Code coverage is published on codecov along with its results and a link to detailed coverage information is attached as a comment to PR.
Test results are published as workflow artifacts in raw format and as HTML test reports, where applicable.
Currently, there are no plans to change the build process, except minor parameter modifications or added features such as release wheel publishing to tt-forge.

## Testing
Testing is performed inside build-and-test.yml workflow as run-tests jobs.
It uses a matrix strategy which means that multiple jobs are created and executed on multiple machines using the same job task.

### Test Matrix
Test matrix is defined as JSON file inside `.github/test_matrix` directory. Currently, only `pr-push-matrix.json` is used.
Each row in the matrix array represents one test that will execute on a specific machine using specified (release) build image.
Example:

```json
  {"runs-on": "n150",   "image": "tracy",  "type": "pykernel"},
  {"runs-on": "n300",   "image": "speedy", "type": "ttrt", "path": "Silicon", "args": "--non-zero", "flags": "run"},
```

#### runs-on
Specify the machine on which the test suite will be executed.
Currently supported runners are:

- n150 - Wormhole 1 chip card
- n300 - Wormhole 2 chip card
- llmbox - Loudbox machine with with 4 N300 cards
- tg - galaxy box
- p150b - Blackhole 1 chip card

> It is expected that list will expand soon as machines with blackhole chip family are added to the runner pool.

#### image
Specify which release build image to use. It can be:

- __speedy__
- __tracy__

Please take a look at the [Builds](#builds) section for a more detailed description of the builds.

#### type
Test type. It is name of the BASH script that executes the test. Scripts are located in `.github/test_scripts` directory and it is possible to create new test types simply
by adding scripts to the directory.

#### path (optional)
This field represents the path.
It is up to the test script how it will use this argument.

#### args (optional)
Specify additional arguments for test execution.
It is up to the test script how it will use this argument.

#### flags (optional)
Additional flags may be used when running tests.
It is up to the test script how it will use this argument.


## Adding New Test
Usually, it is enough to add a single line to the test matrix and your tests will become part of tt-mlir CI.
Here is a checklist of what you should decide before adding it:
- On which TT hardware should your tests run? Put the specific hardware in "runs-on" field. If you want your test to run on multiple hardware types add multiple lines to the matrix, one for each hardware type.
- Are your test run with `ttrt` or `pytest` or any other standard type that other tests also use? Put this decision in "type" field.
- Refer to test script you've put in type for interpretation of `path`, `args`, and `flags` parameters.
> __Each line__ in matrix __MUST__ be __unique__! There is no point to run the same test with same build image on the same type of the hardware.

#### Consider
Here are few things to consider:
- Design your `ttrt` test so it is generated with a `-- check_ttmlir` CMake target.
- For pytest, use pytest test discovery to run all tests in subdirectories. In most cases there is no need for two sets of tests.
- If you want to have separate test reports, do not add additional XML file paths and steps to upload these. Use `test_report_path` because it will be automatically picked up and sent to analytics.
- If separate reports are required, treat them as different tests. Add an additional line to the test matrix.
- If you need to add additional steps to the run-tests job, make sure it's necessary. Typically, it's not a good idea to add additional steps. If there's another way to achieve your goal, use that method instead. This is because each step is executed for each test in the test matrix. When you add additional steps your test might pass, but many other tests will randomly fail.

### Test Type Scripts

Before running test scripts workflow will activate python venv and set a number of usual environment variables that might be usful:

- WORK_DIR - contains repository root.
- BUILD_DIR - contains `build` directory transfered from the build step.
- INSTALL_DIR - contains `install` directory transfered from the build step.
- LD_LIBRARY_PATH is set to `lib` directory inside install, as well as lib directory inside TT_MLIR toolchain.
- SYSTEM_DESC_PATH is set to ttrt generated `system_desc.ttsys` file.
- TT_METAL_HOME is set to `tt-metal` directory containing tt-metal artifacts inside install.
- RUNS_ON machine label as specified by [runs-on](#runs-on) field inside test matrix JSON.
- IMAGE_NAME name of the build image as specified by [image](#image) field inside test matrix JSON.
- TTRT_REPORT_PATH - full path with file name for ttrt report for that specific test.
- TEST_REPORT_PATH - full path with file name for xml test report collected and used in analytics.

Also, soft link to install directory is created inside build directory.

All these variables can be used in test matrix in [path](#path-optional), [args](#args-optional), and [flags](#flags-optional).

### Adding New Test Execution Type

In some cases your test might not fit to any of the existing test types.
It is enough to add file inside `.github/test_scripts` directory.
> Please remember to add execution file flag (`chmod +x <script_name>`) and bash shebang (`#!/bin/bash`) to the test script.

A good practice is to put some comments how script interpret `path`, `args`, and `flags` parameters.

The best way to describe how to design your test script is to look at existing ones. For example `builder.sh`:

```bash
# path: path to pytest test files
# args: pytest marker expression to select tests to run
# flags: "run-ttrt" or predefined additional flags for pytest and ttrt

runttrt=""
TTRT_ARGS=""
PYTEST_ARGS=""
for flag in $3; do
    [[ "$flag" == "run-ttrt" ]] && runttrt=1
    [[ "$flag" == "disable-eth-dispatch" ]] && TTRT_ARGS="$TTRT_ARGS --disable-eth-dispatch"
    [[ "$flag" == "require-opmodel" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-opmodel"
    [[ "$flag" == "require-exact-mesh" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-exact-mesh"
done

pytest $1 -m "$2" $PYTEST_ARGS -v --junit-xml=$TEST_REPORT_PATH
if [[ "$runttrt" == "1" ]]; then
    ttrt run $TTRT_ARGS ttir-builder-artifacts/
    cp run_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
    ttrt run $TTRT_ARGS stablehlo-builder-artifacts/
    cp run_results.json ${TTRT_REPORT_PATH%_*}_stablehlo_${TTRT_REPORT_PATH##*_} || true
fi
```

This script has several types of flags that can be stated concurently `run-ttrt` is used if you want to perform ttrt run after pytest and other
possible flags are additional flags for pytest or ttrt. `args` are used as pytest marker string because it contains spaces.
This test uses TTRT_REPORT_PATH but due to the fact that it has two ttrt runs it inserts its type inside filename.

THe second example is `pytest.sh` script:

```bash
# path: path to pytest test files
# args: additional arguments to pass to pytest
# flags: python packages to install before running tests

if [ -n "$3" ]; then
    eval "pip install $3"
fi
export TT_EXPLORER_GENERATED_MLIR_TEST_DIRS=$BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/perf,$BUILD_DIR/test/python/golden/ttnn
export TT_EXPLORER_GENERATED_TTNN_TEST_DIRS=$BUILD_DIR/test/python/golden/ttnn
pytest -ssv $1 $2 --junit-xml=$TEST_REPORT_PATH
```

Note how it uses eval command to expand bash variables where it is suitable. It also define some additional environment variables using provided ones.
