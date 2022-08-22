import json
from pathlib import Path

import pytest

from eodc_pg_parser.graph import OpenEOProcessGraph
from eodc_pg_parser.pg_schema import (ProcessArgument, ProcessGraph,
                                      ProcessNode, ResultReference)

TEST_DATA_DIR = Path("tests/data/")

# Run the tests across all these process graphs
@pytest.fixture(
    params=[
        "fit_rf_pg_0.json",
        "s2_max_ndvi_global_parameter.json",
        "pg-evi-example.json",
    ]
)
def flat_process_graph(request) -> dict:
    yield json.load(open(TEST_DATA_DIR / request.param, mode="r"))


@pytest.fixture
def nested_process_graph(flat_process_graph) -> dict:
    yield OpenEOProcessGraph._unflatten_raw_process_graph(flat_process_graph)


@pytest.fixture
def openeo_graph(flat_process_graph) -> OpenEOProcessGraph:
    yield OpenEOProcessGraph(pg_data=flat_process_graph)


def test_unflattening(flat_process_graph, nested_process_graph):
    # TODO: Test that flattening an unflattened graph is equivalent to what it started as

    print(nested_process_graph)


def test_data_model_parsing(flat_process_graph, openeo_graph):
    print(openeo_graph)


def test_data_types_explicitly():
    flat_process_graph = json.load(open(TEST_DATA_DIR / "fit_rf_pg_0.json", mode="r"))
    nested_process_graph = OpenEOProcessGraph._unflatten_raw_process_graph(
        flat_process_graph
    )
    parsed_process_graph = OpenEOProcessGraph._parse_datamodel(nested_process_graph)
    assert isinstance(parsed_process_graph, ProcessGraph)
    assert isinstance(parsed_process_graph.process_graph["root"], ProcessNode)
    assert isinstance(
        parsed_process_graph.process_graph["root"].arguments["model"], ProcessArgument
    )
    assert isinstance(
        parsed_process_graph.process_graph["root"].arguments["model"].__root__,
        ResultReference,
    )
    assert isinstance(
        parsed_process_graph.process_graph["root"].arguments["model"].__root__.node,
        ProcessNode,
    )

    print("")
