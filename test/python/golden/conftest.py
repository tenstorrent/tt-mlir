import pytest

def pytest_addoption(parser):
    parser.addoption(
            "--path", action="store", default=".", help="Path to store test artifacts (e.g. flatbuffers and .mlir files)"
    )

@pytest.fixture
def artifact_path(request):
    return request.config.getoption("--path")
