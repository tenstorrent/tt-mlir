# Internal Build Notes / IRD

- When building the runtime we must use Ubuntu 20.04 docker image
  - When making an IRD reservation use `--docker-image
    yyz-gitlab.local.tenstorrent.com:5005/tenstorrent/infra/ird-ubuntu-20-04-amd64:latest`
- You'll have to manaully install a newer version of cmake, at least 3.20, the easiest way to do this is to `pip install cmake` and make sure this one is in your path
- You'll want LLVM installation to persist IRD reservations, you can achieve this by:
  - mkdir /localdev/$USER/ttmlir-toolchain
  - When requesting an IRD use `--volumes /localdev/$USER/ttmlir-toolchain:/opt/ttmlir-toolchain`

## Working with docker images

Components
  - Dockerfile
  - workflow for building docker image
  - project build using docker image

Overview

The [Dockerfile](.github/Dockerfile) describes how to create an image for building the tt-mlir project file. It starts with a supported base image (Ubuntu 20.04) and installs the necessary packages. The purpose of the Docker build is to:

  - Setup build dependencies
  - Prepare the tt-mlir toolchain

During the Docker build, the project is built and tests are run to ensure that everything is set up correctly. If any dependencies are missing, the Docker build will fail.

This process also prepopulates caches for Python packages and the ccache cache in the image, which should make subsequent builds faster.

### Building the docker image using Github Action

Github action workflow [Build and Publish Docker Image](.github/workflows/build-image.yml) builds docker image and uploads it to Github parckages https://github.com/orgs/tenstorrent/packages?repo_name=tt-mlir. Image name is tt-mlir-ubuntu-20-04 and we uses git SHA we build from as tag.

### Building the docker image locally

To test the changes and build the image locally use command
```bash
docker build -f .github/Dockerfile -t tt-mlir-ubuntu-20-04:latest .
```

### Pushing the Docker image to Github

Images build locally can be pushed to Github. We first need to generate PAT token with enabled "write:packages" access
Github -> Settings -> Developer settings -> Personal access tokens -> Generate new token

Authenticate with GitHub Container Registry
```bash
echo "<my-github-pat>" | docker login ghcr.io -u <username> --password-stdin
```

Add tag to built image
```bash
docker tag tt-mlir-ubuntu-20-04:latest ghcr.io/tenstorrent/tt-mlir/tt-mlir-ubuntu-20-04:latest
```

Push
```bash
docker push ghcr.io/tenstorrent/tt-mlir/tt-mlir-ubuntu-20-04:latest
```

### Using the image in Github Action jobs

Github Action workflow [Build in Docker](.github/workflows/docker-build.yml) uses docker container for building
```yaml
    container:
      image: ghcr.io/${{ github.repository }}/tt-mlir-ubuntu-20-04:latest
      options: --user root
```
