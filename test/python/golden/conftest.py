import pytest
from ttrt.common.api import API as ttrt


def pytest_addoption(parser):
    parser.addoption(
        "--path",
        action="store",
        default=".",
        help="Path to store test artifacts (e.g. flatbuffers and .mlir files)",
    )


@pytest.fixture
def artifact_path(request):
    return request.config.getoption("--path")


@pytest.fixture(autouse=True)
def sys_desc():
    """
    Before any tests are run, query the system so the descriptor is always up to date
    """
    ttrt.initialize_apis()
    args = {"--save-artifacts": True}
    ttrt.Query(args=args)()
