# CI

Our CI infrastructure is currently hosted in the cloud. Cloud machines are used and linked as GitHub runners.

CI is triggered by new pull requests and pushes to main (usually when a PR is merged).

CI is designed to automatically collect analytics data for each workflow run, including test results and code coverage. It also publishes the latest release of documentation on GitHub.

## Builds
CI performs the following build jobs:

> - Release build **speedy** - release image optimized for speed
> - Release build **tracy** - release image with runtime trace/debug capabilities including performance measurements
> - Debug build with unit tests and test coverage
> - Clang tidy
> - ...and [Tests](#testing)

The build of tt-mlir is done using the build-tt-mlir-action.
Only the Debug build has a specific implementation because it is also used to run unit tests and collect and publish code coverage data.
Code coverage is published on codecov along with its results, and a link to detailed coverage information is attached as a comment to the PR.
Test results are published as workflow artifacts in raw format and as HTML test reports, where applicable.
Currently, there are no plans to change the build process, except for minor parameter modifications or added features such as release wheel publishing to tt-forge.

## Testing
Testing is performed inside the build-and-test.yml workflow as run-tests jobs.
It uses a matrix strategy, which means that multiple jobs are created and executed on multiple machines using the same job task.

### Test Matrix
The test matrix is defined as a JSON file inside the `.github/test_matrix` directory. Currently, only `pr-push-matrix.json` is used.
Each row in the matrix array represents a test that will execute on a specific machine using a specified (release) build image.
Example:

```json
  { "runs-on": "n150",   "image": "tracy",  "script": "pykernel.sh" },
  { "runs-on": "n300",   "image": "speedy", "script": "ttrt.sh", "args": ["run", "Silicon", "--non-zero"] },
```

#### runs-on
Specifies the machine on which the test suite will be executed.
Currently supported runners are:

- n150 - Wormhole 1 chip card
- n300 - Wormhole 2 chip card
- llmbox - Loudbox machine with 4 N300 cards
- tg - Galaxy box
- p150 - Blackhole 1 chip card

> It is expected that the list will expand soon as machines with blackhole chip family are added to the runner pool.

#### image
Specifies which release build image to use. It can be:

- **speedy**
- **tracy**

Please take a look at the [Builds](#builds) section for a more detailed description of the builds.

#### script
Test type. It is the name of the BASH script that executes the test. Scripts are located in the `.github/test_scripts` directory,
and it is possible to create new test types simply by adding scripts to the directory.

#### args (optional)
This field represents the arguments for the script. This can be omitted, a string, or a JSON array.

#### reqs (optional)
Specifies additional requirements for test execution.
These arguments are passed as the REQUIREMENTS environment variable to the test script.

### Using JSON arrays
The **runs-on** and **image** fields can be passed as JSON arrays. With arrays, one can define a test to execute on multiple machines and images.
Examples:

```json
{ "runs-on": ["n150","n300"],
    "image": ["speedy","tracy"],
        "script": "unit" }
```

## Adding New Test
Usually, it is enough to add a single line to the test matrix and your tests will become part of the tt-mlir CI.
Here is a checklist of what you should decide before adding it:
- On which TT hardware should your tests run? Put the specific hardware in the "runs-on" field.
- Do your tests run with `ttrt` or `pytest` or any other standard type that other tests also use? Put this decision in the "script" field.
- Refer to the test script you've put in the type for interpretation of `args` and `reqs` parameters.
> **Each line** in the matrix **MUST** be **unique**! There is no point in running the same test with the same build image on the same type of hardware.

#### Consider
Here are a few things to consider:
- Design your `ttrt` test so it is generated with a `-- check_ttmlir` CMake target. These will be generated at compile time and will be available for test jobs.
- For pytest, use pytest test discovery to run all tests in subdirectories. In most cases, there is no need for two sets of tests.
- If you want to have separate test reports, do not add additional XML file paths and steps to upload these.
  Use `${TTRT_REPORT_PATH}` (for ttrt JSON files) or `${TEST_REPORT_PATH}` (for JUnit XML) because it will be automatically picked up and sent to analytics.
- If separate reports are required, treat them as different tests. Add an additional line to the test matrix.
  You can use a construct from this example: `cp run_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_}`

### Adding New Test Script
To design a new test script, add new files to `.github/test_scripts`
> Make sure you set the execution flag for the new test script file (`chmod +x <file>`)!

Before running test scripts, the workflow will activate the default TT-MLIR Python venv and set a number of useful environment variables:
- WORK_DIR - set to repo root
- BUILD_DIR - set to build artifacts
- INSTALL_DIR - set to install artifacts
- LD_LIBRARY_PATH - set to install artifacts `lib` and toolchain `lib` directories
- SYSTEM_DESC_PATH - set to `system_desc.ttsys` system descriptor file generated by `ttrt`
- TT_METAL_HOME - set to `tt-metal` install directory
- RUNS_ON - machine label script runs on
- IMAGE_NAME - name of the image script runs on
- RUN_ID - id of workflow run, [see below](#downloading-artifacts).

Also, a soft link is created inside the build directory to the install directory.

> Please make sure you implement cleanup logic inside your script and leave the script with the repo in the same state as before execution!

A good practice is to put some comments on how the script interprets arguments (and requirements if applicable).

For example `builder.sh`:

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

This script has several types of flags that can be stated concurrently. Arguments are parsed as `run-ttrt` and other
possible flags for pytest or ttrt. This test uses TTRT_REPORT_PATH, but due to the fact that it has two ttrt runs, it inserts its type inside the filename.

The second example is `pytest.sh` script:

```bash
if [ -n "$REQUIREMENTS" ]; then
    eval "pip install $REQUIREMENTS"
fi
export TT_EXPLORER_GENERATED_MLIR_TEST_DIRS=$BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/perf,$BUILD_DIR/test/python/golden/ttnn
export TT_EXPLORER_GENERATED_TTNN_TEST_DIRS=$BUILD_DIR/test/python/golden/ttnn
pytest -ssv "$@" --junit-xml=$TEST_REPORT_PATH
```
This script uses $REQUIREMENTS to specify additional wheels to be installed.
Note how it uses the eval command to expand bash variables where suitable. It also defines some additional environment variables using the provided ones.

#### Downloading Artifacts

If you need to download artifacts (e.g., wheels) from a workflow run, you can use the following command:

```bash
gh run download $RUN_ID --repo tenstorrent/tt-mlir --name <artifact_name>
```

This command downloads the specified artifact to the current directory. You can specify a different download location using the `--dir <directory>` option.

To download multiple artifacts matching a pattern, use the `--pattern` option instead of `--name`:

```bash
gh run download $RUN_ID --repo tenstorrent/tt-mlir --pattern "tt_*.whl"
```

**Note:** When using `--pattern`, artifacts are downloaded into separate subdirectories, even when `--dir` is specified.


## CI Run (under the hood)

Test runs are prepared in the `prepare-run` job when the input test matrix is transformed into a job test matrix that will be used for test runs.
All jobs are grouped based on `runs-on` and `image` fields and then split (and balanced) into several runs based on test durations and target total time.
This is done to make efficient use of resources because there are many tests that last for just several seconds while preparation can take ~4 minutes.
So, tests are run in batches in the `Run Test` step with clear separation and summary. Lists of tests are displayed at the beginning, and one can search for `test <number>`
(number ranges from 1 to total number of tests) when needing to see test flow and test results for a particular test. Also, it is possible during development
to comment out tests in the JSON file of [Test Matrix](#test-matrix) using the `#` character in a development branch and make test runs much faster, but
please do not forget to remove comments when a PR is created or finalized.
Test durations are collected after each push to main, and these are automatically used on each subsequent PR, Push, and other runs.
