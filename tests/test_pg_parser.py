import datetime
import json
import logging

import numpy as np
import pendulum
import pyproj
import pytest
from pydantic import ValidationError

from openeo_pg_parser_networkx import OpenEOProcessGraph, Process
from openeo_pg_parser_networkx.pg_schema import *
from tests.conftest import TEST_DATA_DIR, TEST_NODE_KEY

logger = logging.getLogger(__name__)


def test_full_parse(process_graph_path):
    parsed_graph_from_file = OpenEOProcessGraph.from_file(process_graph_path)
    parsed_graph_from_json = OpenEOProcessGraph.from_json(
        json.dumps(json.load(open(process_graph_path)))
    )
    assert isinstance(parsed_graph_from_file, OpenEOProcessGraph)
    assert parsed_graph_from_file == parsed_graph_from_json

    # Dry-run to_callable after parsing
    mock_process_registry = {
        process_id: Process({}, lambda process_id: logger.debug(process_id), "predefined")
        for process_id in parsed_graph_from_file.required_processes
    }
    callable = parsed_graph_from_file.to_callable(mock_process_registry)

    parsed_graph_from_file.plot()


def test_named_parameters():
    # Create a mock process that verifies named parameters
    def mock_process(*args, named_parameters=None, **kwargs):
        assert named_parameters is not None
        assert named_parameters == {"test_param": "test_value"}
        return "success"

    # Create process graph
    pg_data = {
        "process_graph": {
            "test_node": {"process_id": "mock_process", "arguments": {}, "result": True}
        }
    }

    # Create process graph
    parsed_graph = OpenEOProcessGraph(pg_data)

    # Create process registry with our mock process
    process_registry = {"mock_process": Process({}, mock_process, "predefined")}

    # Execute process graph with named parameters
    callable = parsed_graph.to_callable(process_registry)
    result = callable(named_parameters={"test_param": "test_value"})

    assert result == "success"


def test_function_generation():
    from openeo_pg_parser_networkx.utils import generate_curve_fit_function

    flat_process_graph = json.load(open(TEST_DATA_DIR / "graphs" / "all_math.json"))
    parsed_graph = OpenEOProcessGraph.from_json(json.dumps(flat_process_graph))

    result = generate_curve_fit_function(parsed_graph)(None, None)

    expected_result = 0.49253470118  # This was calculated by hand

    assert np.isclose(result, expected_result, atol=1e-10)


def test_fit_curve_parse():
    flat_process_graph = json.load(open(TEST_DATA_DIR / "graphs" / "fit_curve.json"))
    parsed_graph = OpenEOProcessGraph.from_json(json.dumps(flat_process_graph))
    assert isinstance(parsed_graph, OpenEOProcessGraph)

    # Dry-run to_callable after parsing
    mock_process_registry = {
        process_id: Process({}, lambda process_id: logger.debug(process_id), "predefined")
        for process_id in parsed_graph.required_processes
    }
    callable = parsed_graph.to_callable(mock_process_registry)

    parsed_graph.plot()


def test_aggregate_temporal_period_parse():
    flat_process_graph = json.load(open(TEST_DATA_DIR / "graphs" / "aggregate.json"))
    parsed_graph = OpenEOProcessGraph.from_json(json.dumps(flat_process_graph))
    assert isinstance(parsed_graph, OpenEOProcessGraph)

    # Dry-run to_callable after parsing
    mock_process_registry = {
        process_id: Process({}, lambda process_id: logger.debug(process_id), "predefined")
        for process_id in parsed_graph.required_processes
    }
    callable = parsed_graph.to_callable(mock_process_registry)

    parsed_graph.plot()


def test_from_json_constructor():
    flat_process_graph = json.load(open(TEST_DATA_DIR / "graphs" / "fit_rf_pg.json"))
    parsed_graph = OpenEOProcessGraph.from_json(json.dumps(flat_process_graph))
    assert isinstance(parsed_graph, OpenEOProcessGraph)


def test_data_types_explicitly():
    flat_process_graph = json.load(open(TEST_DATA_DIR / "graphs" / "fit_rf_pg.json"))
    nested_process_graph = OpenEOProcessGraph._unflatten_raw_process_graph(
        flat_process_graph
    )
    parsed_process_graph = OpenEOProcessGraph._parse_datamodel(nested_process_graph)
    assert isinstance(parsed_process_graph, ProcessGraph)
    assert isinstance(parsed_process_graph.process_graph["root"], ProcessNode)
    assert isinstance(
        parsed_process_graph.process_graph["root"].arguments["data"], ResultReference
    )
    assert isinstance(
        parsed_process_graph.process_graph["root"].arguments["data"].node,
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
        ProcessGraph.model_validate(pg)
        .process_graph[TEST_NODE_KEY]
        .arguments["spatial_extent"]
    )
    assert isinstance(parsed_arg, BoundingBox)
    assert isinstance(parsed_arg.crs, str)
    assert parsed_arg.crs == pyproj.CRS.from_user_input('EPSG:2025').to_wkt()


def test_pydantic_loading():
    test_extent = {'west': 0, 'east': 10, 'south': 0, 'north': 10}
    test_bb = BoundingBox(**test_extent)
    assert test_bb.crs == DEFAULT_CRS


def test_bounding_box_no_crs(get_process_graph_with_args):
    pg = get_process_graph_with_args(
        {'spatial_extent': {'west': 0, 'east': 10, 'south': 0, 'north': 10}}
    )
    parsed_arg = (
        ProcessGraph.model_validate(pg)
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
        ProcessGraph.model_validate(pg).process_graph[TEST_NODE_KEY].arguments[
            "spatial_extent"
        ]


def test_string_validation(get_process_graph_with_args):
    '''
    During the pydantic 2 update, we found that some special strings get parsed
    to non-string values ('t' to True, 'f' to False, etc.)

    Check that every incoming string stays a string by default
    '''

    test_args = {
        'arg_t': 't',
        'arg_f': 'f',
        'arg_str': 'arg_123_str',
        'arg_int': '123',
        'arg_float': '123.4',
    }

    pg = get_process_graph_with_args(test_args)

    # Parse indirectly to check if model validation is strict and does not type coerce
    parsed_graph = OpenEOProcessGraph(pg_data=pg)

    # Parse directly to check if strict model validation works seperately
    parsed_args = [
        ProcessGraph.model_validate(pg, strict=True)
        .process_graph[TEST_NODE_KEY]
        .arguments[arg_name]
        for arg_name in test_args.keys()
    ]

    resolved_kwargs = parsed_graph.nodes[0][1]['resolved_kwargs'].items()

    assert all([isinstance(resolved_kwarg, str) for _, resolved_kwarg in resolved_kwargs])

    assert all([isinstance(parsed_arg, str) for parsed_arg in parsed_args])


@pytest.mark.parametrize(
    "specific_graph,expected_nodes",
    [path for path in zip((TEST_DATA_DIR / "graphs").glob('none_*.json'), [4, 4, 4])],
)
def test_none_parameter(specific_graph, expected_nodes):
    with open(specific_graph) as fp:
        pg_data = json.load(fp=fp)

    parsed_graph = OpenEOProcessGraph(pg_data=pg_data)
    assert len(parsed_graph.nodes) == expected_nodes


def test_bounding_box_int_crs(get_process_graph_with_args):
    pg = get_process_graph_with_args(
        {'spatial_extent': {'west': 0, 'east': 10, 'south': 0, 'north': 10, 'crs': 4326}}
    )
    parsed_arg = (
        ProcessGraph.model_validate(pg)
        .process_graph[TEST_NODE_KEY]
        .arguments["spatial_extent"]
    )
    assert isinstance(parsed_arg, BoundingBox)
    assert isinstance(parsed_arg.crs, str)
    assert parsed_arg.crs == DEFAULT_CRS


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
        ProcessGraph.model_validate(pg)
        .process_graph[TEST_NODE_KEY]
        .arguments["geometries"]
    )
    assert isinstance(parsed_arg, get_args(GeoJson))


@pytest.mark.skip(
    reason="Not passing because of https://github.com/developmentseed/geojson-pydantic/issues/92"
)
def test_geojson_parsing():
    with pytest.raises(ValidationError):
        should_not_parse = GeoJson.model_validate(['vh', 'vv'])


def test_jobid(get_process_graph_with_args):
    argument = {'job_id': 'jb-4da83382-8f8e-4153-8961-e15614b04185'}
    pg = get_process_graph_with_args(argument)
    parsed_arg = (
        ProcessGraph.model_validate(pg).process_graph[TEST_NODE_KEY].arguments["job_id"]
    )
    assert isinstance(parsed_arg, JobId)

def test_temporal_intervals(get_process_graph_with_args):
    argument1 = {
        'temporal_intervals': [
            ['1990-01-01T12:00:00', '20:00:00'],
            ['1995-01-30', '2000'],
            ['1995-01-30', None],
            ['15:00:00', '1990-01-01T20:00:00'],
            ['2022-09-01 00:00:00+00:00', '2023-01-01 00:00:00+00:00'],
        ]
    }
    pg = get_process_graph_with_args(argument1)
    parsed_intervals = (
        ProcessGraph.model_validate(pg)
        .process_graph[TEST_NODE_KEY]
        .arguments["temporal_intervals"]
    )
    assert isinstance(parsed_intervals, TemporalIntervals)

    first_interval = parsed_intervals[0]
    second_interval = parsed_intervals[1]
    third_interval = parsed_intervals[2]
    fourth_interval = parsed_intervals[3]

    assert isinstance(first_interval, TemporalInterval)
    assert isinstance(first_interval.start, DateTime)
    assert isinstance(first_interval.end, DateTime)
    assert first_interval.end.root == first_interval.start.root.add(hours=8)

    assert isinstance(second_interval, TemporalInterval)
    assert isinstance(second_interval.start, Date)
    assert isinstance(second_interval.end, Year)

    assert isinstance(third_interval, TemporalInterval)
    assert isinstance(third_interval.start, Date)
    assert third_interval.end is None

    assert isinstance(fourth_interval, TemporalInterval)
    assert isinstance(fourth_interval.start, DateTime)
    assert isinstance(fourth_interval.end, DateTime)


def test_invalid_temporal_intervals():
    with pytest.raises(ValidationError):
        TemporalInterval.model_validate(['1990-01-01T12:00:00', '11:00:00'])
    with pytest.raises(ValidationError):
        TemporalInterval.model_validate([None, None])
    with pytest.raises(ValidationError):
        TemporalInterval.model_validate(['15:00:00', '1990-01-01T20:00:00', '11:00:00'])
    with pytest.raises(ValidationError):
        TemporalInterval.model_validate(['1990-01-01T20:00:00'])
    with pytest.raises(ValidationError):
        TemporalInterval.model_validate([None, '13:00:00'])
    with pytest.raises(ValidationError):
        TemporalInterval.model_validate(['13:00:00', None])
    with pytest.raises(ValidationError):
        TemporalInterval.model_validate(['13:00:00', '14:00:00'])


def test_duration(get_process_graph_with_args):
    argument = {'duration': 'P1Y1M1DT2H'}
    pg = get_process_graph_with_args(argument)
    parsed_arg = (
        ProcessGraph.model_validate(pg).process_graph[TEST_NODE_KEY].arguments["duration"]
    )
    assert isinstance(parsed_arg, Duration)
    assert isinstance(parsed_arg.root, datetime.timedelta)

    assert parsed_arg.to_numpy() == np.timedelta64(
        pendulum.parse(argument["duration"]).as_timedelta()
    )


def test_datetime(get_process_graph_with_args):
    argument_valid = {'datetime': '1975-05-21T22:00:00'}
    pg = get_process_graph_with_args(argument_valid)
    parsed_arg = (
        ProcessGraph.model_validate(pg).process_graph[TEST_NODE_KEY].arguments["datetime"]
    )
    assert isinstance(parsed_arg, DateTime)
    assert isinstance(parsed_arg.root, datetime.datetime)
    assert parsed_arg.to_numpy() == np.datetime64(argument_valid["datetime"])

    with pytest.raises(ValidationError):
        DateTime.model_validate('21-05-1975T22:00:00')


def test_date(get_process_graph_with_args):
    argument_valid = {'date': '1975-05-21'}
    pg = get_process_graph_with_args(argument_valid)
    print(pg)
    parsed_arg = (
        ProcessGraph.model_validate(pg).process_graph[TEST_NODE_KEY].arguments["date"]
    )
    print(parsed_arg)
    assert isinstance(parsed_arg, Date)
    assert isinstance(parsed_arg.root, datetime.datetime)
    assert parsed_arg.to_numpy() == np.datetime64(argument_valid["date"])

    with pytest.raises(ValidationError):
        DateTime.model_validate('21-05-1975')
        DateTime.model_validate('22:00:80')


def test_year(get_process_graph_with_args):
    argument_valid = {'year': '1975'}
    pg = get_process_graph_with_args(argument_valid)
    parsed_arg = (
        ProcessGraph.model_validate(pg).process_graph[TEST_NODE_KEY].arguments["year"]
    )
    assert isinstance(parsed_arg, Year)
    assert isinstance(parsed_arg.root, datetime.datetime)
    assert parsed_arg.to_numpy() == np.datetime64(argument_valid["year"])

    with pytest.raises(ValidationError):
        DateTime.model_validate('75')
        DateTime.model_validate('0001')
        DateTime.model_validate('22:00:80')


def test_time(get_process_graph_with_args):
    argument_valid = {'time': '22:00:00'}
    pg = get_process_graph_with_args(argument_valid)
    parsed_arg = (
        ProcessGraph.model_validate(pg).process_graph[TEST_NODE_KEY].arguments["time"]
    )
    assert isinstance(parsed_arg, Time)
    assert isinstance(parsed_arg.root, pendulum.Time)

    with pytest.raises(NotImplementedError):
        parsed_arg.to_numpy()

    with pytest.raises(ValidationError):
        DateTime.model_validate('22:00:80')
        DateTime.model_validate('0001')
