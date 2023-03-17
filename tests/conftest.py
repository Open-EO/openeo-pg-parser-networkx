import json
from pathlib import Path
from typing import Callable

import pytest

from openeo_pg_parser_networkx import Process, ProcessRegistry

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


@pytest.fixture
def process_registry() -> ProcessRegistry:
    registry = ProcessRegistry(wrap_funcs=[])

    _max = lambda data, dimension=None, ignore_nodata=True, **kwargs: data.max(
        dim=dimension, skipna=ignore_nodata, keep_attrs=True
    )

    max_spec = json.load(open(TEST_DATA_DIR / "max.json"))

    registry["_max"] = Process(spec=max_spec, implementation=_max, namespace="predefined")

    return registry
