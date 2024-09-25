
import pytest
import os

@pytest.fixture(autouse=True)
def configure_logging_for_each_test(request):
    import logging
    test_name = request.node.name
    log_file = f"{test_name}.log"
    
    # Debugging to ensure fixture is invoked
    print(f"Setting up logging for {test_name}")
    
    
    # Basic configuration for logging
    logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode="w", 
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.debug(f"Starting test: {test_name}")


