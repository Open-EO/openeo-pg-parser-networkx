import datetime
import json

import numpy as np
import pendulum
import pyproj
import pytest
from pydantic import ValidationError

from openeo_pg_parser_networkx import OpenEOProcessGraph
from openeo_pg_parser_networkx.pg_schema import *
from tests.conftest import TEST_DATA_DIR, TEST_NODE_KEY


def test_full_parse(process_graph_path):
    parsed_graph_from_file = OpenEOProcessGraph.from_file(process_graph_path)
    parsed_graph_from_json = OpenEOProcessGraph.from_json(
        json.dumps(json.load(open(process_graph_path)))
    )
    assert isinstance(parsed_graph_from_file, OpenEOProcessGraph)
    assert parsed_graph_from_file == parsed_graph_from_json

    # Dry-run to_callable after parsing
    mock_process_registry = {
        process_id: lambda process_id: print(process_id)
        for process_id in parsed_graph_from_file.required_processes
    }
    callable = parsed_graph_from_file.to_callable(mock_process_registry)

    parsed_graph_from_file.plot()


def test_from_json_constructor():
    flat_process_graph = json.load(open(TEST_DATA_DIR / "graphs" / "fit_rf_pg_0.json"))
    parsed_graph = OpenEOProcessGraph.from_json(json.dumps(flat_process_graph))
    assert isinstance(parsed_graph, OpenEOProcessGraph)


def test_data_types_explicitly():
    flat_process_graph = json.load(open(TEST_DATA_DIR / "graphs" / "fit_rf_pg_0.json"))
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
    pg = get_process_graph_with_args(
        {
            'spatial_extent': {
                'west': 0,
                'east': 10,
                'south': 0,
                'north': 10,
                'crs': "EPSG:2025",
            }
        }
    )
    parsed_arg = (
        ProcessGraph.parse_obj(pg)
        .process_graph[TEST_NODE_KEY]
        .arguments["spatial_extent"]
    )
    assert isinstance(parsed_arg, BoundingBox)
    assert isinstance(parsed_arg.crs, str)
    assert parsed_arg.crs == pyproj.CRS.from_user_input('EPSG:2025').to_wkt()


def test_bounding_box_no_crs(get_process_graph_with_args):
    pg = get_process_graph_with_args(
        {'spatial_extent': {'west': 0, 'east': 10, 'south': 0, 'north': 10, 'crs': ""}}
    )
    parsed_arg = (
        ProcessGraph.parse_obj(pg)
        .process_graph[TEST_NODE_KEY]
        .arguments["spatial_extent"]
    )
    assert isinstance(parsed_arg, BoundingBox)
    assert isinstance(parsed_arg.crs, str)
    assert parsed_arg.crs == DEFAULT_CRS


def test_bounding_box_with_faulty_crs(get_process_graph_with_args):
    pg = get_process_graph_with_args(
        {
            'spatial_extent': {
                'west': 0,
                'east': 10,
                'south': 0,
                'north': 10,
                'crs': "hello",
            }
        }
    )
    with pytest.raises(pyproj.exceptions.CRSError):
        ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments[
            "spatial_extent"
        ]


@pytest.mark.skip(
    reason="Not passing because of https://github.com/developmentseed/geojson-pydantic/issues/92"
)
def test_geojson(get_process_graph_with_args):
    from typing import get_args

    # TODO: Generate arbitrary GeoJSONs for testing using something like this hypothesis extension: https://github.com/mapbox/hypothesis-geojson
    argument = {
        'geometries': {
            "type": "FeatureCollection",
            "features": [
                {
                    "id": 0,
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [102.0, 0.5]},
                    "properties": {"prop0": "value0"},
                }
            ],
        }
    }
    pg = get_process_graph_with_args(argument)
    parsed_arg = (
        ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["geometries"]
    )
    assert isinstance(parsed_arg, get_args(GeoJson))


@pytest.mark.skip(
    reason="Not passing because of https://github.com/developmentseed/geojson-pydantic/issues/92"
)
def test_geojson_parsing():
    with pytest.raises(ValidationError):
        should_not_parse = GeoJson.parse_obj(['vh', 'vv'])


def test_jobid(get_process_graph_with_args):
    argument = {'job_id': 'jb-4da83382-8f8e-4153-8961-e15614b04185'}
    pg = get_process_graph_with_args(argument)
    parsed_arg = (
        ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["job_id"]
    )
    assert isinstance(parsed_arg, JobId)


def test_output_format(get_process_graph_with_args):
    #         regex=r"(gtiff|GTiff|geotiff|GeoTiff|Netcdf|NetCDF|netcdf|json)"

    valid_file_formats = [
        "gtiff",
        "GTiff",
        "netCDF",
        "Netcdf",
        "netcdf",
        "json",
        "geotiff",
        "GeoTiff",
    ]
    arguments = [{'output_format': v} for v in valid_file_formats]
    for argument in arguments:
        pg = get_process_graph_with_args(argument)
        parsed_arg = (
            ProcessGraph.parse_obj(pg)
            .process_graph[TEST_NODE_KEY]
            .arguments["output_format"]
        )
        assert isinstance(parsed_arg, OutputFormat)

    invalid_file_formats = ["yo", "pdf"]
    arguments = [{'output_format': v} for v in invalid_file_formats]
    for argument in arguments:
        pg = get_process_graph_with_args(argument)
        parsed_arg = (
            ProcessGraph.parse_obj(pg)
            .process_graph[TEST_NODE_KEY]
            .arguments["output_format"]
        )
        assert not isinstance(parsed_arg, OutputFormat)


def test_temporal_intervals(get_process_graph_with_args):
    argument1 = {
        'temporal_intervals': [
            ['1990-01-01T12:00:00', '20:00:00'],
            ['1995-01-30', '2000'],
            ['1995-01-30', '2000'],
        ]
    }
    pg = get_process_graph_with_args(argument1)
    parsed_intervals = (
        ProcessGraph.parse_obj(pg)
        .process_graph[TEST_NODE_KEY]
        .arguments["temporal_intervals"]
    )
    assert isinstance(parsed_intervals, TemporalIntervals)

    first_interval = parsed_intervals[0]
    second_interval = parsed_intervals[1]

    assert isinstance(first_interval, TemporalInterval)
    assert isinstance(first_interval.start, DateTime)
    assert isinstance(first_interval.end, Time)

    assert isinstance(second_interval, TemporalInterval)
    assert isinstance(second_interval.start, Date)
    assert isinstance(second_interval.end, Year)


def test_duration(get_process_graph_with_args):
    argument = {'duration': 'P1Y1M1DT2H'}
    pg = get_process_graph_with_args(argument)
    parsed_arg = (
        ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["duration"]
    )
    assert isinstance(parsed_arg, Duration)
    assert isinstance(parsed_arg.__root__, pendulum.Duration)
    with pytest.raises(NotImplementedError):
        parsed_arg.to_numpy()


def test_datetime(get_process_graph_with_args):
    argument_valid = {'datetime': '1975-05-21T22:00:00'}
    pg = get_process_graph_with_args(argument_valid)
    parsed_arg = (
        ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["datetime"]
    )
    assert isinstance(parsed_arg, DateTime)
    assert isinstance(parsed_arg.__root__, datetime.datetime)
    assert parsed_arg.to_numpy() == np.datetime64(argument_valid["datetime"])

    with pytest.raises(ValidationError):
        DateTime.parse_obj('21-05-1975T22:00:00')


def test_date(get_process_graph_with_args):
    argument_valid = {'date': '1975-05-21'}
    pg = get_process_graph_with_args(argument_valid)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["date"]
    assert isinstance(parsed_arg, Date)
    assert isinstance(parsed_arg.__root__, datetime.datetime)

    with pytest.raises(NotImplementedError):
        parsed_arg.to_numpy()

    with pytest.raises(ValidationError):
        DateTime.parse_obj('21-05-1975')
        DateTime.parse_obj('22:00:80')


def test_year(get_process_graph_with_args):
    argument_valid = {'year': '1975'}
    pg = get_process_graph_with_args(argument_valid)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["year"]
    assert isinstance(parsed_arg, Year)
    assert isinstance(parsed_arg.__root__, datetime.datetime)
    assert parsed_arg.to_numpy() == np.datetime64(argument_valid["year"])

    with pytest.raises(ValidationError):
        DateTime.parse_obj('75')
        DateTime.parse_obj('0001')
        DateTime.parse_obj('22:00:80')


def test_time(get_process_graph_with_args):
    argument_valid = {'time': '22:00:00'}
    pg = get_process_graph_with_args(argument_valid)
    parsed_arg = ProcessGraph.parse_obj(pg).process_graph[TEST_NODE_KEY].arguments["time"]
    assert isinstance(parsed_arg, Time)
    assert isinstance(parsed_arg.__root__, pendulum.Time)

    with pytest.raises(NotImplementedError):
        parsed_arg.to_numpy()

    with pytest.raises(ValidationError):
        DateTime.parse_obj('22:00:80')
        DateTime.parse_obj('0001')
