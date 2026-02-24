# Working with Docker Images

Components:

- Dockerfile
- Workflow for building Docker image
- Project build using Docker image

## Overview

We use docker images to prepare the project environment, install dependencies, tooling and prebuild toolchain.
Project builds four docker images:

- Base image `tt-mlir-base-ubuntu-24-04` [Dockerfile.base](.github/Dockerfile.base)
- CI image `tt-mlir-ci-ubuntu-24-04` [Dockerfile.ci](.github/Dockerfile.ci)
- Base IRD image `tt-mlir-base-ird-ubuntu-24-04`[Dockerfile.ird](.github/Dockerfile.ird)
- IRD image `tt-mlir-ird-ubuntu-24-04` [Dockerfile.ird](.github/Dockerfile.ird)

Base image starts with a supported base image (Ubuntu 22.04) and installs dependencies for project build. From there, we build the CI image that contains the prebuild toolchain and is used in CI to shorten the build time. The IRD image contains dev tools such as GDB, vim and ssh which are used in IRD environments.

During the CI Docker build, the project is built and tests are run to ensure that everything is set up correctly. If any dependencies are missing, the Docker build will fail.

## Using the Docker Image

Here is a typical command to run the latest developer (ird) docker image:

```bash
sudo docker run -it -d --rm \
  --name my-docker \
  --cap-add ALL \
  --device /dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-24-04:latest bash
```

> Special attention should be paid to flags:
> - `--device /dev/tenstorrent/0:/dev/tenstorrent/0`: this is required to map
>   the hardware device into the container. For machines with multiple devices,
>   this flag can be specified multiple times or adjusted with the appropriate
>   device number.
> - `-v /dev/hugepages:/dev/hugepages` / `-v /dev/hugepages-1G:/dev/hugepages-1G`:
>   this is required to map the hugepages volume into the container. For more
>   information on hugepages, please refer to the [Getting Started Guide](./getting-started#step-4-set-up-hugepages).
>
> The base or CI image can also be used in the same way, but the IRD image is
> recommended for development.

## Using the Docker Image via IRD (_Internal Developers Only_)

Internally we use a tool called IRD.  As part of your `reserve` command, you
can specify the docker image to use:

```bash
ird reserve \
  --docker-image ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-24-04:latest
```

See `ird reserve --help` for more information on the `reserve` command. Typical
ird usage might look like:

```bash
# list machine availability
ird list-machines

# reserve a machine
ird reserve \
  --volumes /localdev/$USER:/localdev/$USER \
  --docker-image ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-24-04:latest \
  --timeout 720 \
  wormhole_b0 \
  --machine [MACHINE_NAME]

# list your currently reserved machines
ird list

# connect to the first reserved machine
ird connect-to 1

# release the first reserved machine
ird release 1
```

## Building the Docker Image using GitHub Actions

The GitHub Actions workflow [Build and Publish Docker Image](.github/workflows/build-image.yml) builds the Docker images and uploads them to GitHub Packages at [https://github.com/orgs/tenstorrent/packages?repo_name=tt-mlir](https://github.com/orgs/tenstorrent/packages?repo_name=tt-mlir). We use the git SHA we build from as the tag.

## Building the Docker Image Locally

To test the changes and build the image locally, use the following command:

```bash
docker build -f .github/Dockerfile.base -t ghcr.io/tenstorrent/tt-mlir/tt-mlir-base-ubuntu-24-04:latest .
docker build -f .github/Dockerfile.ci -t ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-24-04:latest .
docker build -f .github/Dockerfile.ird --build-arg FROM_IMAGE=base -t ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-base-ubuntu-24-04:latest .
docker build -f .github/Dockerfile.ird --build-arg FROM_IMAGE=ci -t ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-24-04:latest .
```

## Using the Image in GitHub Actions Jobs

The GitHub Actions workflow [Build in Docker](.github/workflows/docker-build.yml) uses a Docker container for building:

```yaml
    container:
      image: ghcr.io/${{ github.repository }}/tt-mlir-ci-ubuntu-24-04:latest
      options: --user root
```
