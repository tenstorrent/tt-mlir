# Working with Docker Images

Components:
  - Dockerfile
  - Workflow for building Docker image
  - Project build using Docker image

## Overview

We use docker images to prepare project enviroment, install dependancies, tooling and prebuild toolchain.
Project builds four docker images:

Base image `tt-mlir-base-ubuntu-22-04` [Dockerfile.base](.github/Dockerfile.base)
CI image `tt-mlir-ci-ubuntu-22-04` [Dockerfile.ci](.github/Dockerfile.ci)
Base IRD image `tt-mlir-base-ird-ubuntu-22-04`[Dockerfile.ird](.github/Dockerfile.ird)
IRD image `tt-mlir-ird-ubuntu-22-04` [Dockerfile.ird](.github/Dockerfile.ird)

Base image starts with a supported base image (Ubuntu 22.04) and installs dependancies for project build. From there we build CI image that contains prebuild toolcahin and is used in CI to shoten the build time. IRD image contain dev tools like GDB, vim etc and shh and are use in IRD enviroments.

During the CI Docker build, the project is built and tests are run to ensure that everything is set up correctly. If any dependencies are missing, the Docker build will fail.

## Building the Docker Image using GitHub Actions

The GitHub Actions workflow [Build and Publish Docker Image](.github/workflows/build-image.yml) builds the Docker images and uploads them to GitHub Packages at https://github.com/orgs/tenstorrent/packages?repo_name=tt-mlir. We use the git SHA we build from as the tag.

## Building the Docker Image Locally

To test the changes and build the image locally, use the following command:
```bash
docker build -f .github/Dockerfile.base -t ghcr.io/tenstorrent/tt-mlir/tt-mlir-base-ubuntu-22-04:latest .
docker build -f .github/Dockerfile.ci -t ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-22-04:latest .
docker build -f .github/Dockerfile.ird -build-args FROM_IMAGE=base -t ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-base-ubuntu-22-04:latest .
docker build -f .github/Dockerfile.ird -build-args FROM_IMAGE=ci -t ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-22-04:latest .
```

## Using the Image in GitHub Actions Jobs

The GitHub Actions workflow [Build in Docker](.github/workflows/docker-build.yml) uses a Docker container for building:
```yaml
    container:
      image: ghcr.io/${{ github.repository }}/tt-mlir-ci-ubuntu-22-04:latest
      options: --user root
```
