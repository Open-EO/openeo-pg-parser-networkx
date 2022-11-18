from pathlib import Path
from typing import Callable, Dict

import pytest

TEST_DATA_DIR = Path("tests/data/")
TEST_NODE_KEY = "test_node"

test_pg_graphs = [pg_path for pg_path in (TEST_DATA_DIR / "graphs").glob('**/*.json')]


@pytest.fixture
def get_process_graph_with_args() -> Callable:
    """Function to generate a one-node process graph json and inject an arbitrary argument into."""

    def _get_process_graph_with_args(arguments) -> dict:
        graph = {
            "process_graph": {
                "test_node": {
                    "process_id": "test_process_id",
                    "arguments": arguments,
                    "result": True,
                }
            }
        }
        return graph

    return _get_process_graph_with_args


# Run the tests across all these process graphs
@pytest.fixture(params=test_pg_graphs)
def process_graph_path(request) -> Path:
    return request.param
