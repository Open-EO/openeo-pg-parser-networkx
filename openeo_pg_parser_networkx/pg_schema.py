from __future__ import annotations

import datetime
import json
import logging
from enum import Enum
from re import match
from typing import Any, Optional, Union
from uuid import UUID, uuid4

import numpy as np
import pendulum
import pyproj
from geojson_pydantic import (
    Feature,
    FeatureCollection,
    GeometryCollection,
    MultiPolygon,
    Polygon,
)
from pydantic import BaseModel, Extra, Field, ValidationError, constr, validator
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

# TODO: Move this to a proper settings object for this repo and use in tests, possibly using pydantic settings: https://pydantic-docs.helpmanual.io/usage/settings/
DEFAULT_CRS = pyproj.CRS.from_user_input("EPSG:4326").to_wkt()

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
    "Time",
    "TemporalInterval",
    "TemporalIntervals",
    "GeoJson",
    "JobId",
    "OutputFormat",
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
    arguments: dict[
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
                TemporalInterval,
                TemporalIntervals,
                # GeoJson, disable while https://github.com/developmentseed/geojson-pydantic/issues/92 is open
                Time,
                float,
                str,
                bool,
                list,
                dict,
            ]
        ],
    ]

    def __str__(self):
        return json.dumps(self.dict(), indent=4)


class ProcessGraph(BaseModel, extra=Extra.forbid):
    process_graph: dict[str, ProcessNode]
    uid: UUID = Field(default_factory=uuid4)


class PGEdgeType(str, Enum):
    ResultReference = "result_reference"
    Callback = "callback"


def parse_crs(v) -> pyproj.CRS:
    if v is None or v.strip() == "":
        return DEFAULT_CRS
    else:
        try:
            # Check that the crs can be parsed and store as WKT
            crs_obj = pyproj.CRS.from_user_input(v)
            return crs_obj.to_wkt()
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
    crs: Optional[str]

    # validators
    _parse_crs: classmethod = crs_validator('crs')

    @property
    def polygon(self) -> Polygon:
        """"""
        return Polygon(
            [
                (self.west, self.south),
                (self.west, self.north),
                (self.east, self.north),
                (self.east, self.south),
            ]
        )


class Date(BaseModel):
    __root__: datetime.datetime

    @validator("__root__", pre=True)
    def validate_time(cls, value: Any) -> Any:
        if (
            isinstance(value, str)
            and len(value) <= 11
            and match(r"[0-9]{4}[-/][0-9]{2}[-/][0-9]{2}T?", value)
        ):
            return pendulum.parse(value)
        raise ValidationError("Could not parse `Date` from input.")

    def to_numpy(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__root__.__repr__()


class DateTime(BaseModel):
    __root__: datetime.datetime

    @validator("__root__", pre=True)
    def validate_time(cls, value: Any) -> Any:
        if isinstance(value, str) and match(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T?[0-9]{2}:[0-9]{2}:?([0-9]{2})?Z?", value
        ):
            return pendulum.parse(value)
        raise ValidationError("Could not parse `DateTime` from input.")

    def to_numpy(self):
        return np.datetime64(self.__root__)

    def __repr__(self):
        return self.__root__.__repr__()


class Time(BaseModel):
    __root__: pendulum.Time

    @validator("__root__", pre=True)
    def validate_time(cls, value: Any) -> Any:
        if (
            isinstance(value, str)
            and len(value) >= 5
            and len(value) <= 9
            and match(r"[0-9]{2}:[0-9]{2}:?([0-9]{2})?Z?", value)
        ):
            return pendulum.parse(value).time()
        raise ValidationError("Could not parse `Time` from input.")

    def to_numpy(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__root__.__repr__()


class Year(BaseModel):
    __root__: datetime.datetime

    @validator("__root__", pre=True)
    def validate_time(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) <= 4 and match(r"^\d{4}$", value):
            return pendulum.parse(value)
        raise ValidationError("Could not parse `Year` from input.")

    def to_numpy(self):
        return np.datetime64(self.__root__)

    def __repr__(self):
        return self.__root__.__repr__()


class Duration(BaseModel):
    __root__: pendulum.Duration

    @validator("__root__", pre=True)
    def validate_time(cls, value: Any) -> Any:
        if isinstance(value, str) and match(
            r"P[0-9]*Y?[0-9]*M?[0-9]*D?T?[0-9]*H?[0-9]*M?[0-9]*S?", value
        ):
            return pendulum.parse(value)
        raise ValidationError("Could not parse `Duration` from input.")

    def to_numpy(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__root__.__repr__()


class TemporalInterval(BaseModel):
    __root__: list[Union[Year, Date, DateTime, Time]]

    @property
    def start(self):
        return self.__root__[0]

    @property
    def end(self):
        return self.__root__[1]

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]


class TemporalIntervals(BaseModel):
    __root__: list[TemporalInterval]

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item) -> TemporalInterval:
        return self.__root__[item]


GeoJson = Union[FeatureCollection, Feature, GeometryCollection, MultiPolygon, Polygon]
# The GeoJson spec (https://www.rfc-editor.org/rfc/rfc7946.html#ref-GJ2008) doesn't
# have a crs field anymore and recommends assuming it to be EPSG:4326, so we do the same.


class JobId(BaseModel):
    __root__: str = Field(
        regex=r"(eodc-jb-|jb-)[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}"
    )


class OutputFormat(BaseModel):
    __root__: str = Field(regex=r"(?i)(gtiff|geotiff|netcdf|json)")


ResultReference.update_forward_refs()
ProcessNode.update_forward_refs()
