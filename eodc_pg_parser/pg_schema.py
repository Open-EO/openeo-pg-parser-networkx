from __future__ import annotations

import json
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Extra, Field, constr, validator
import pyproj

import logging

logger = logging.getLogger(__name__)

# TODO: Move this to a proper settings object for this repo and use in tests, possibly using pydantic settings: https://pydantic-docs.helpmanual.io/usage/settings/
DEFAULT_CRS = pyproj.CRS.from_user_input("EPSG:4326")


# This controls what is imported when calling `from pg_schema import *`, just a shortcut to import all types.
__all__ = [
    "ResultReference",
    "ParameterReference",
    "ProcessNode",
    "ProcessGraph",
    "PGEdgeType",
    "BoundingBox",
    "Year",
    "Date",
    "DateTime",
    "Duration",
    "Features",
    "GeoJson",
    "JobId",
    "OutputFormat",
    "Time",
    "TemporalInterval",
    "TemporalIntervals",
    "URI",
    "DEFAULT_CRS",
]


class ResultReference(BaseModel, extra=Extra.forbid):
    from_node: str
    node: ProcessNode


class ParameterReference(BaseModel, extra=Extra.forbid):
    from_parameter: str


class ProcessNode(BaseModel, arbitrary_types_allowed=True):
    process_id: constr(regex=r'^\w+$')
    namespace: Optional[Optional[str]] = None
    result: Optional[bool] = False
    description: Optional[Optional[str]] = None
    arguments: Dict[
        str,
        Optional[
            Union[
                ResultReference,
                ParameterReference,
                ProcessGraph,
                BoundingBox,
                JobId,
                OutputFormat,
                Year,
                Date,
                DateTime,
                Duration,
                GeoJson,
                Time,
                TemporalInterval,
                TemporalIntervals,
                URI,
                float,
                str,
                bool,
                List,
                Dict,
            ]
        ],
    ]

    def __str__(self):
        return json.dumps(self.dict(), indent=4)


class ProcessGraph(BaseModel, extra=Extra.forbid):
    process_graph: Dict[str, ProcessNode]
    uid: UUID = Field(default_factory=uuid4)


class PGEdgeType(str, Enum):
    ResultReference = "result_reference"
    Callback = "callback"


def parse_crs(v) -> pyproj.CRS:
    if v is None or v.strip() == "":
        return DEFAULT_CRS
    else:
        try:
            return pyproj.CRS.from_user_input(v)
        except pyproj.exceptions.CRSError as e:
            logger.error(f"Provided CRS {v} could not be parsed, defaulting to EPSG:4326")
            raise e


def crs_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True, pre=True, always=True)
    validator_func = decorator(parse_crs)
    return validator_func


class BoundingBox(BaseModel, arbitrary_types_allowed=True):
    west: float
    east: float
    north: float
    south: float
    base: Optional[float]
    height: Optional[float]
    crs: Optional[pyproj.CRS]

    # validators
    _parse_crs: classmethod = crs_validator('crs')


class Year(
    BaseModel
):  # a more general option would be: [0-9]{4}, but I assume we want years from 1900 to 2100?
    __root__: str = Field(regex=r"(19|20)[0-9]{2}", max_length=4)


class Date(BaseModel):
    __root__: str = Field(regex=r"[0-9]{4}[-/][0-9]{2}[-/][0-9]{2}T?", max_length=11)


class DateTime(BaseModel):
    __root__: str = Field(
        regex=r"[0-9]{4}-[0-9]{2}-[0-9]{2}T?[0-9]{2}:[0-9]{2}:?([0-9]{2})?Z?",
        min_length=15,
        max_length=20,
    )


class Duration(BaseModel):
    __root__: str = Field(regex=r"P[0-9]*Y?[0-9]*M?[0-9]*D?T?[0-9]*H?[0-9]*M?[0-9]*S?")


class Features(BaseModel):
    id: Optional[str]
    type: str
    geometry: Dict
    properties: Optional[Dict]


class GeoJson(BaseModel, arbitrary_types_allowed=True):
    type: str
    features: List[Features]
    crs: Optional[pyproj.CRS]

    # validators
    _parse_crs: classmethod = crs_validator('crs')


class JobId(BaseModel):
    __root__: str = Field(
        regex=r"(eodc-jb-|jb-)[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}"
    )


class OutputFormat(BaseModel):
    __root__: str = Field(
        regex=r"(gtiff|GTiff|geotiff|GeoTiff|Netcdf|NetCDF|netcdf|json)"
    )


class Time(BaseModel):
    __root__: str = Field(
        regex=r"[0-9]{2}:[0-9]{2}:?([0-9]{2})?Z?", min_length=5, max_length=9
    )


class TemporalInterval(BaseModel):
    __root__: List[Union[Year, Date, DateTime, Time]]


class TemporalIntervals(BaseModel):
    __root__: List[TemporalInterval]


class URI(BaseModel):
    __root__: str = Field(regex=r"[a-zA-Z]*[\:\/\.][.]*")


ResultReference.update_forward_refs()
ProcessNode.update_forward_refs()
