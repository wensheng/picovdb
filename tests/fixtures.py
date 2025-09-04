from pathlib import Path
import pytest


@pytest.fixture(scope="function")
def cleanup_test_db():
    """Fixture to remove test database files after test execution."""
    # Define paths
    base_name = "picovdb"
    db_path = Path(f"{base_name}.vecs.npy")
    meta_path = Path(f"{base_name}.meta.json")
    ids_path = Path(f"{base_name}.ids.json")

    # Setup: Ensure files don't exist before test
    db_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    ids_path.unlink(missing_ok=True)

    yield base_name  # Let the test run, optionally yield the base name

    # Teardown: Cleanup after test
    db_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    ids_path.unlink(missing_ok=True)
