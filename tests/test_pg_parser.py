import json
from pathlib import Path

import pytest
import pyproj

from eodc_pg_parser.graph import OpenEOProcessGraph
from eodc_pg_parser.pg_schema import *
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

def test_bounding_box():
    argument = {'__root__': {'west': 0, 'east': 10, 'south': 0, 'north': 10, 'crs': "EPSG:2025"}}
    parsed_arg = ProcessArgument(**argument).__root__
    assert isinstance(parsed_arg, BoundingBox)
    assert isinstance(parsed_arg.crs, pyproj.CRS)
    assert parsed_arg.crs == 'EPSG:2025'

def test_bounding_box_without_crs():
    argument = {'__root__': {'west': 0, 'east': 10, 'south': 0, 'north': 10, 'crs': "hello"}}
    parsed_arg = ProcessArgument(**argument).__root__
    assert isinstance(parsed_arg, BoundingBox)
    assert isinstance(parsed_arg.crs, pyproj.CRS)
    assert parsed_arg.crs == DEFAULT_CRS

def test_geojson():
    # TODO: Generate arbitrary GeoJSONs for testing using something like this hypothesis extension: https://github.com/mapbox/hypothesis-geojson
    argument = {'__root__': {
        "type": "FeatureCollection", 
        "features": [{
            "id": 0, 
            "type": "Feature", 
            "geometry": {"type": "Point", "coordinates": [102.0, 0.5]}, 
            "properties": {"prop0": "value0"}
            }],
        "crs": "EPSG:2025"
        }}
    parsed_arg = ProcessArgument(**argument).__root__
    assert isinstance(parsed_arg, GeoJson)
    assert parsed_arg.crs == 'EPSG:2025'

def test_geojson_without_crs():
    argument = {'__root__': {
        "type": "FeatureCollection",
        "features": [{
            "id": 0, 
            "type": "Feature", 
            "geometry": {"type": "Point", "coordinates": [102.0, 0.5]}, 
            "properties": {"prop0": "value0"}
            }]
        }}
    parsed_arg = ProcessArgument(**argument).__root__
    assert isinstance(parsed_arg, GeoJson)
    assert parsed_arg.crs == DEFAULT_CRS

def test_jobid():
    argument = {'__root__': 'jb-4da83382-8f8e-4153-8961-e15614b04185'}
    parsed_arg = ProcessArgument(**argument).__root__
    assert isinstance(parsed_arg, JobId) 

def test_outputformat():
    argument = {'__root__': 'GTiff'}
    parsed_arg = ProcessArgument(**argument).__root__
    assert isinstance(parsed_arg, OutputFormat)

def test_uri():
    argument = {'__root__': 'http://uri.com/'}
    parsed_arg = ProcessArgument(**argument).__root__
    assert isinstance(parsed_arg, URI)

def test_temporal_intervall():
    argument = {'__root__': ['1990-01-01T12:00:00', '20:00:00']}
    parsed_arg = ProcessArgument(**argument).__root__
    assert isinstance(parsed_arg, TemporalInterval)
    assert isinstance(parsed_arg.__root__[0], DateTime)
    assert isinstance(parsed_arg.__root__[1], Time)

    argument = {'__root__': ['1990-01-01', '2000']}
    parsed_arg = ProcessArgument(**argument).__root__
    assert isinstance(parsed_arg, TemporalInterval)
    assert isinstance(parsed_arg.__root__[0], Date)
    assert isinstance(parsed_arg.__root__[1], Year)

def test_duration():
    argument = {'__root__': 'P1Y1M1DT2H'}
    parsed_arg = ProcessArgument(**argument).__root__
    assert isinstance(parsed_arg, Duration)
