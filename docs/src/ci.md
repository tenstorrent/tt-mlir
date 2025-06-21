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
Each row in the matrix array represents one test that will execute on a specific machine.
Example:

```yaml
 {runs-on: n150,   name: "run",  suite: "runtime_debug",     image: "tracy",  type: "ttrt",    path: "Silicon", flags: "--non-zero", container-options: "--device /dev/tenstorrent/0"},
 {runs-on: llmbox, name: "perf", suite: "perf",              image: "tracy",  type: "ttrt",    path: "Silicon/TTNN/llmbox/perf", container-options: "--device /dev/tenstorrent/0 --device /dev/tenstorrent/1 --device /dev/tenstorrent/2 --device /dev/tenstorrent/3"},
 {runs-on: n150,   name: "perf", suite: "explorer",          image: "tracy",  type: "pytest",  path: "tools/explorer/test/run_tests.py", container-options: "--device /dev/tenstorrent/0"},
```

#### runs-on
Specify the machine on which the test suite will be executed.
Currently supported runners are:

- N150
- N300
- NXX0 - either N150 or N300
- llmbox
- tg - galaxy box

> It is expected that list will expand soon as machines with blackhole chip family are added to the runner pool.

#### name
"name" has historic origins in its name.
In reality it is the type of test to perform:

- __run__ - perform functional run, or just run tests
- __perf__ - collect performance data (and send them to analytics)

#### path
This field represents the path inside the tt-mlir repository where your tests resides.
For `ttrt` test this is the relative path for generated mlir files inside the build/test/ttmlir directory.
For pytest the path is relative to the repository root.

#### suite
This is the actual test name.

#### image
Specify which release build image to use. It can be:

- __speedy__
- __tracy__

Please take a look at the [Builds](#builds) section for a more detailed description of the builds.

#### type
Specify the type of test run. Currently supported:
- __pytest__ - run python tests using pytest
- __ttrt__ - run tests using `ttrt` tool
- __unit__ - run unit tests
- __builder__ - run builder tests and execute generated flatbuffers iff `run-ttrt` flag is set
- __ttnn_standalone__ - run `ttnn_standalone` sanity test
- __pykernel__ - run `pykernel` tests and runtime demo.

#### flags (optional)
Additional flags may be used when running tests. These are passed to `ttrt` or pytest as an additional parameter.

#### container-options (optional)
Each test runs in a docker container and this option specifies docker container options.
It is mostly used to map TT hardware device to a docker container (for example: `"--device /dev/tenstorrent/0"`).
If no value is passed, the default value will be used (`"--device /dev/tenstorrent"`)

### Adding New Test
Usually, it is enough to add a single line to the test matrix and your tests will become part of tt-mlir CI.
Here is a checklist of what you should decide before adding it:
- On which TT hardware should your tests should run? Put the specific hardware in "runs-on" field or `NXX0` if you don't care. If you want your test to run on multiple hardware types add multiple lines to the matrix, one for each hardware type.
- Are your test run with `ttrt` or pytest? Put this decision in "type" field.
- Does your test generate performance report? If it does put name as "perf". If not put name as "run".
- Use creativity and name your test. Write result of your hard intellectual work inside "suite" field.
> __Each line__ in matrix __MUST__ be __unique__! Check if it is. If it is not, use more of your creative and intellectual energy to create better (at least different) name for "suite" field.

#### Consider
Here are few things to consider:
- Design your `ttrt` test so it is generated with a `-- check_ttmlir` CMake target.
- For pytest, use pytest test discovery to run all tests in subdirectories. In most cases there is no need for two sets of tests.
- If you want to have separate test reports, do not add additional XML file paths and steps to upload these. Use `test_report_path` because it will be automatically picked up and sent to analytics.
- If separate reports are required, treat them as different tests. Add an additional line to the test matrix.
- If you need to add additional steps to the run-tests job, make sure it's necessary. Typically, it's not a good idea to add additional steps. If there's another way to achieve your goal, use that method instead. This is because each step is executed for each test in the test matrix. When you add additional steps your test might pass, but many other tests will randomly fail.
