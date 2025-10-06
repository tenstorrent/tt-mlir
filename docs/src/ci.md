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
Each row in the matrix array represents test that will execute on a specific machine using specified (release) build image.
Example:

```json
  { "runs-on": "n150",   "image": "tracy",  "script": "pykernel.sh" },
  { "runs-on": "n300",   "image": "speedy", "script": "ttrt.sh", "args": ["run", "Silicon", "--non-zero"] },
```

#### runs-on
Specify the machine on which the test suite will be executed.
Currently supported runners are:

- n150 - Wormhole 1 chip card
- n300 - Wormhole 2 chip card
- llmbox - Loudbox machine with with 4 N300 cards
- tg - galaxy box
- p150 - Blackhole 1 chip card

> It is expected that list will expand soon as machines with blackhole chip family are added to the runner pool.

#### image
Specify which release build image to use. It can be:

- __speedy__
- __tracy__

Please take a look at the [Builds](#builds) section for a more detailed description of the builds.

#### script
Test type. It is name of the BASH script that executes the test. Scripts are located in `.github/test_scripts` directory and it is possible to create new test types simply
by adding scripts to the directory.

#### args (optional)
This field represents the arguments for the script. This can be ommited, string or json array.

#### reqs (optional)
Specify additional requirements for test execution.
This arguments are passed as REQUIREMENTS environment variable to test script.

### Using JSON arrays
**runs-on** and **image** fields can be passed as JSON array. With arrays one can define test to execute on multiple machines and images.
Examples:

```json
{ "runs-on": ["n150","n300"],
    "image": ["speedy","tracy"],
        "script": "unit" }
```

## Adding New Test
Usually, it is enough to add a single line to the test matrix and your tests will become part of tt-mlir CI.
Here is a checklist of what you should decide before adding it:
- On which TT hardware should your tests run? Put the specific hardware in "runs-on" field. If you want your test to run on multiple hardware types add multiple lines to the matrix, one for each hardware type.
- Are your test run with `ttrt` or `pytest` or any other standard type that other tests also use? Put this decision in "script" field.
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

A good practice is to put some comments how script interpret arguments (and requirements if applicable).

The best way to describe how to design your test script is to look at existing ones. For example `builder.sh`:

```bash
# arg $1: path to pytest test files
# arg $2: pytest marker expression to select tests to run
# arg $3: "run-ttrt" or predefined additional flags for pytest and ttrt

runttrt=""
TTRT_ARGS=""
PYTEST_ARGS=""

[[ "$RUNS_ON" != "n150" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-exact-mesh"
[[ "$RUNS_ON" == "p150" ]] && TTRT_ARGS="$TTRT_ARGS --disable-eth-dispatch"

for flag in $3; do
    [[ "$flag" == "run-ttrt" ]] && runttrt=1
    [[ "$flag" == "require-opmodel" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-opmodel"
done

pytest "$1" -m "$2" $PYTEST_ARGS -v --junit-xml=$TEST_REPORT_PATH
if [[ "$runttrt" == "1" ]]; then
    ttrt run $TTRT_ARGS ttir-builder-artifacts/
    cp run_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
    ttrt run $TTRT_ARGS stablehlo-builder-artifacts/
    cp run_results.json ${TTRT_REPORT_PATH%_*}_stablehlo_${TTRT_REPORT_PATH##*_} || true
fi
```

This script has several types of flags that can be stated concurently. Argument is parsed as `run-ttrt` and other
possible flags for pytest or ttrt. This test uses TTRT_REPORT_PATH but due to the fact that it has two ttrt runs it inserts its type inside filename.

The second example is `pytest.sh` script:

```bash
if [ -n "$REQUIREMENTS" ]; then
    eval "pip install $REQUIREMENTS"
fi
export TT_EXPLORER_GENERATED_MLIR_TEST_DIRS=$BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/perf,$BUILD_DIR/test/python/golden/ttnn
export TT_EXPLORER_GENERATED_TTNN_TEST_DIRS=$BUILD_DIR/test/python/golden/ttnn
pytest -ssv "$@" --junit-xml=$TEST_REPORT_PATH
```
This script uses $REQUIREMENTS to specify additional wheel to be installed.
Note how it uses eval command to expand bash variables where it is suitable. It also define some additional environment variables using provided ones.

## CI Run (under the hood)

Test run is prepared in `prepare-run` job when input test matrix is transformed into real test matrix that will be used for test runs.
All jobs are grouped based on `runs-on` and `image` fields and then split (and balanced) to several runs based on test durations and target total time.
This is done to make efficient use of resources because there are many tests that last for just several seconds while preparation can take ~4 minutes.
So, tests are run in batches in `Run Test` step with clear separation and summary. List of tests are displayed on the beginning and one can search for `test <number>`
(number range from 1 to total number of tests) when in need to see test flow and test results for particular test. Also, it is possible during development
to comment out tests in JSON file of [Test Matrix](#test-matrix) using `#` character in development branch and make test runs much faster but
please do not forget to remove comments when PR is created or finalized.
Test durations are collected after each push to main and these are automatically used on each subsequent PR, Push, and other runs.
