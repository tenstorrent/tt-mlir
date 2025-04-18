# `CI`

Our CI infrastructure is currently hosted on cloud. Cloud machines are used and linked as GitHub runners.

CI is triggered by new pull request and on push into main (usually when PR is merged).

CI is designed to automatically collect analytics data for each workflow run, including test results and code coverage. It will also r publish newest release of documentation on GitHub.

## Builds
CI performs following build jobs:

> - Release build "__speedy__" - release image optimized for speed.
> - Release build "__tracy__" - release image with runtime trace/debug capabilities including performance measurments.
> - Debug build with unit tests and test coverage.
> - CLang tidy
> - ...and [Tests](#testing)

Build of tt-mlir is done using build-tt-mlir-action.
Only Debug build has specific implementation because it is also used to run unit tests and collect and publish code coverage data.
Code coverage is published on codecov and its results with link to detailed coverage information is attached as a comment to PR.
Test results are published as workflow artifacts in raw format and as HTML test reports, where applicable.
Currently, there are no plans to change build process, except minor parameter modifications or added features such as release wheel publishing to tt-forge.

## Testing
Testing is performed inside build-and-test.yml workflow as run-tests jobs.
It is uses matrix strategy which means that multiple jobs are created and executed on multiple machines using same job task.

### Test Matrix
Each row in the matrix array represent one test that will execute on specific machine.
Example:

```yaml
 {runs-on: n150,   name: "run",  suite: "runtime_debug",     image: "tracy",  type: "ttrt",    path: "Silicon", flags: "--non-zero", container-options: "--device /dev/tenstorrent/0"},
 {runs-on: n300,   name: "run",  suite: "async",             image: "speedy", type: "ttrt",    path: "Silicon/TTNN", flags: "--non-zero --enable-async-ttnn", container-options: "--device /dev/tenstorrent/0"},
 {runs-on: llmbox, name: "perf", suite: "perf",              image: "tracy",  type: "ttrt",    path: "Silicon/TTNN/llmbox/perf", container-options: "--device /dev/tenstorrent/0 --device /dev/tenstorrent/1 --device /dev/tenstorrent/2 --device /dev/tenstorrent/3"},
 {runs-on: n150,   name: "perf", suite: "explorer",          image: "tracy",  type: "pytest",  path: "tools/explorer/test/run_tests.py", container-options: "--device /dev/tenstorrent/0"},
```

#### runs-on
Specify machine on which test suite will be executed.
Currently supported runners are:

- N150
- N300
- NXX0 - either N150 or N300
- llmbox
- tg - galaxy box

> It is expected that list will expand soon as machines with blackhole chip family are added to the runner pool.

#### name
"name" has historic origins in its name.
In reallity it is the type of test to perform:

- __run__ - perform functional run, or just run tests
- __perf__ - collect performance data (and send them to analytics)

#### path
This field represents the path inside tt-mlir repository where your tests resides.
For ttrt test this is relative path for generated mlir files inside build/test/ttmlir directory.
For pytest it is path relative to repository root.

#### suite
This is the actual test name.

#### image
Specify which release build image to use. It can be:

- __speedy__
- __tracy__

Please take a look at [Builds](#builds) section for more detailed description of the builds.

#### type
Specify type of test run. Currently supported:
- __pytest__ - run python tests using pytest
- __ttrt__ - run tests using ttrt tool
- __unit__ - run unit tests

#### flags (optional)
These are additional flags used when running tests. Passed to ttrt or pytest as additional parameter.

#### container-options
Each test runs in docker container and this option specifies docker container options.
It is mostly used to map TT hardware device to docker container (for example: `"--device /dev/tenstorrent/0"`).

### Adding New Test
Usually, it is enough to add single line to the test matrix and your tests will become part of tt-mlir CI.
Here is checklist of what you should decide before adding it:
- On which TT hardware your tests should run? Put specific hardware in "runs-on" field or `NXX0` if you don't care. If you want your test to run on multiple hardware types add multiple lines to the matrix one for each hardware type.
- Are your test run with ttrt or pytest? Put this decision in "type" field.
- Does your test generate performance report? If it does put name as "perf". If not put name as "run".
- Use creativity and name your test. Write result of your hard intellectual work inside "suite" field.
> __Each line__ in matrix __MUST__ be __unique__! Check if it is. If it is not, use more of your creative and intellectual energy to create better (at least different) name for "suite" field.

#### Consider
Here are few things to consider:
- Design your ttrt test so they are generated with `-- check_ttmlir` target
- For pytest, use pytest test discovery to run all test in subdirectories. In most cases there is no need for two set of tests.
- If there is, for example, you want to have separated test reports, do not add additional xml file paths and steps to upload these. Use `test_report_path` becuase it will be automatically picked up and sent to analytics.
- If you insist that these should have separate reports, assume they are different tests. Add additional line to test matrix.
- Thinking about adding additional step to run-tests job? Think more. Usually, that is not a good idea. If there is alternative, use that alternative. This is because each step is executed for each test in test matrix. When you add additional steps your test might pass but many other tests will randomly fail. Don't ask how I know. It is the path that you _should not_ try walking on.
