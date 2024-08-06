# Internal Build Notes / IRD

- When building the runtime we must use Ubuntu 20.04 docker image
  - When making an IRD reservation use `--docker-image
    yyz-gitlab.local.tenstorrent.com:5005/tenstorrent/infra/ird-ubuntu-20-04-amd64:latest`
- You'll have to manaully install a newer version of cmake, at least 3.20, the easiest way to do this is to `pip install cmake` and make sure this one is in your path
- You'll want LLVM installation to persist IRD reservations, you can achieve this by:
  - mkdir /localdev/$USER/ttmlir-toolchain
  - When requesting an IRD use `--volumes /localdev/$USER/ttmlir-toolchain:/opt/ttmlir-toolchain`

## Working with Docker Images

Components:
  - Dockerfile
  - Workflow for building Docker image
  - Project build using Docker image

Overview:

The [Dockerfile](.github/Dockerfile) describes how to create an image for building the tt-mlir project file. It starts with a supported base image (Ubuntu 20.04) and installs the necessary packages. The purpose of the Docker build is to:

  - Set up build dependencies
  - Prepare the tt-mlir toolchain

During the Docker build, the project is built and tests are run to ensure that everything is set up correctly. If any dependencies are missing, the Docker build will fail.

This process also prepopulates caches for Python packages and the ccache cache in the image, which should make subsequent builds faster.

### Building the Docker Image using GitHub Actions

The GitHub Actions workflow [Build and Publish Docker Image](.github/workflows/build-image.yml) builds the Docker image and uploads it to GitHub Packages at https://github.com/orgs/tenstorrent/packages?repo_name=tt-mlir. The image name is tt-mlir-ubuntu-20-04, and we use the git SHA we build from as the tag.

### Building the Docker Image Locally

To test the changes and build the image locally, use the following command:
```bash
docker build -f .github/Dockerfile -t tt-mlir-ubuntu-20-04:latest .
```

### Pushing the Docker Image to GitHub

Images built locally can be pushed to GitHub. First, we need to generate a PAT token with the "write:packages" access enabled. Go to GitHub -> Settings -> Developer settings -> Personal access tokens -> Generate new token.

Authenticate with GitHub Container Registry:
```bash
echo "<my-github-pat>" | docker login ghcr.io -u <username> --password-stdin
```

Add a tag to the built image:
```bash
docker tag tt-mlir-ubuntu-20-04:latest ghcr.io/tenstorrent/tt-mlir/tt-mlir-ubuntu-20-04:latest
```

Push the image:
```bash
docker push ghcr.io/tenstorrent/tt-mlir/tt-mlir-ubuntu-20-04:latest
```

### Using the Image in GitHub Actions Jobs

The GitHub Actions workflow [Build in Docker](.github/workflows/docker-build.yml) uses a Docker container for building:
```yaml
    container:
      image: ghcr.io/${{ github.repository }}/tt-mlir-ubuntu-20-04:latest
      options: --user root
```
