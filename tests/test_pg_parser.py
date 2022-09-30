import json
from pathlib import Path
from typing import Dict

import pytest
import pyproj

from eodc_pg_parser.graph import OpenEOProcessGraph
from eodc_pg_parser.pg_schema import *

TEST_DATA_DIR = Path("tests/data/")
TEST_NODE_KEY = "test_node"

@pytest.fixture
def get_process_graph_with_args():
    """Function to generate a one-node process graph json and inject an arbitrary argument into."""
    def _get_process_graph_with_args(arguments) -> Dict:
        graph = {"process_graph": {
                        "test_node": {
                            "process_id": "test_process_id",
                            "arguments": arguments,
                            "result": True
                        }
                    }
                }
        return graph

    return _get_process_graph_with_args

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
        parsed_process_graph.process_graph["root"].arguments["model"], ResultReference
    )
    assert isinstance(
        parsed_process_graph.process_graph["root"].arguments["model"].node,
        ProcessNode,
    )

def test_bounding_box(get_process_graph_with_args):
    pg = get_process_graph_with_args({'spatial_extent': {'west': 0, 'east': 10, 'south': 0, 'north': 10, 'crs': "EPSG:2025"}})
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["spatial_extent"]
    assert isinstance(parsed_arg, BoundingBox)
    assert isinstance(parsed_arg.crs, pyproj.CRS)
    assert parsed_arg.crs == 'EPSG:2025'

def test_bounding_box_without_crs(get_process_graph_with_args):
    pg = get_process_graph_with_args({'spatial_extent': {'west': 0, 'east': 10, 'south': 0, 'north': 10, 'crs': "hello"}})
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["spatial_extent"]
    assert isinstance(parsed_arg, BoundingBox)
    assert isinstance(parsed_arg.crs, pyproj.CRS)
    assert parsed_arg.crs == DEFAULT_CRS

def test_geojson(get_process_graph_with_args):
    # TODO: Generate arbitrary GeoJSONs for testing using something like this hypothesis extension: https://github.com/mapbox/hypothesis-geojson
    argument = {'geometries': {
        "type": "FeatureCollection", 
        "features": [{
            "id": 0, 
            "type": "Feature", 
            "geometry": {"type": "Point", "coordinates": [102.0, 0.5]}, 
            "properties": {"prop0": "value0"}
            }],
        "crs": "EPSG:2025"
        }}
    pg = get_process_graph_with_args(argument)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["geometries"]
    assert isinstance(parsed_arg, GeoJson)
    assert parsed_arg.crs == 'EPSG:2025'

def test_geojson_without_crs(get_process_graph_with_args):
    argument = {'geometries': {
        "type": "FeatureCollection", 
        "features": [{
            "id": 0, 
            "type": "Feature", 
            "geometry": {"type": "Point", "coordinates": [102.0, 0.5]}, 
            "properties": {"prop0": "value0"}
            }],
        "crs": ""
        }}
    pg = get_process_graph_with_args(argument)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["geometries"]
    assert isinstance(parsed_arg, GeoJson)
    assert parsed_arg.crs == DEFAULT_CRS

def test_jobid(get_process_graph_with_args):
    argument = {'job_id': 'jb-4da83382-8f8e-4153-8961-e15614b04185'}
    pg = get_process_graph_with_args(argument)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["job_id"]
    assert isinstance(parsed_arg, JobId) 

def test_output_format(get_process_graph_with_args):
    argument = {'output_format': 'GTiff'}
    pg = get_process_graph_with_args(argument)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["output_format"]
    assert isinstance(parsed_arg, OutputFormat)

def test_uri(get_process_graph_with_args):
    argument = {'uri': 'http://uri.com/'}
    pg = get_process_graph_with_args(argument)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["uri"]
    assert isinstance(parsed_arg, URI)

def test_temporal_interval(get_process_graph_with_args):
    argument1 = {'temporal_interval': ['1990-01-01T12:00:00', '20:00:00']}
    pg = get_process_graph_with_args(argument1)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["temporal_interval"]
    assert isinstance(parsed_arg, TemporalInterval)
    assert isinstance(parsed_arg.__root__[0], DateTime)
    assert isinstance(parsed_arg.__root__[1], Time)

    argument2 = {'temporal_interval': ['1990-01-01T12:00:00', '20:00:00']}
    pg = get_process_graph_with_args(argument2)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["temporal_interval"]
    assert isinstance(parsed_arg, TemporalInterval)
    assert isinstance(parsed_arg.__root__[0], DateTime)
    assert isinstance(parsed_arg.__root__[1], Time)

def test_duration(get_process_graph_with_args):
    argument = {'duration': 'P1Y1M1DT2H'}
    pg = get_process_graph_with_args(argument)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["duration"]
    assert isinstance(parsed_arg, Duration)

