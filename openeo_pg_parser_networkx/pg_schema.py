from __future__ import annotations

import datetime
import json
import logging
from enum import Enum
from re import match
from typing import Annotated, Any, List, Optional, Union
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
from pydantic import (
    BaseModel,
    Extra,
    Field,
    RootModel,
    StringConstraints,
    ValidationError,
    constr,
    field_validator,
    model_validator,
    validator,
)
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
    "DEFAULT_CRS",
]


class ResultReference(BaseModel, extra=Extra.forbid):
    from_node: str
    node: ProcessNode


class ParameterReference(BaseModel, extra=Extra.forbid):
    from_parameter: str


class ProcessNode(BaseModel, arbitrary_types_allowed=True):
    process_id: Annotated[str, StringConstraints(pattern=r'^\w+$')]

    namespace: Optional[Optional[str]] = None
    result: Optional[bool] = False
    description: Optional[Optional[str]] = None
    arguments: dict[
        str,
        Annotated[
            Union[
                ResultReference,
                ParameterReference,
                ProcessGraph,
                BoundingBox,
                JobId,
                Year,
                Date,
                DateTime,
                Duration,
                TemporalInterval,
                TemporalIntervals,
                # GeoJson, disable while https://github.com/developmentseed/geojson-pydantic/issues/92 is open
                Time,
                float,
                bool,
                list,
                dict,
                str,
            ],
            Field(union_mode='left_to_right'),
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
    if not isinstance(v, int) and (v is None or v.strip() == ""):
        return DEFAULT_CRS
    else:
        try:
            # Check that the crs can be parsed and store as WKT
            crs_obj = pyproj.CRS.from_user_input(v)
            return crs_obj.to_wkt()
        except pyproj.exceptions.CRSError as e:
            logger.error(f"Provided CRS {v} could not be parsed, defaulting to EPSG:4326")
            raise e


def crs_validator(field: Union[str, int]) -> classmethod:
    decorator = validator(field, allow_reuse=True, pre=True, always=True)
    validator_func = decorator(parse_crs)
    return validator_func


class BoundingBox(BaseModel, arbitrary_types_allowed=True):
    west: float
    east: float
    north: float
    south: float
    base: Optional[float] = None
    height: Optional[float] = None
    crs: Optional[Union[str, int]] = None

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


class Date(RootModel):
    root: datetime.datetime

    @field_validator("root", mode="before")
    def validate_time(cls, value: Any) -> Any:
        if (
            isinstance(value, str)
            and len(value) <= 11
            and match(r"[0-9]{4}[-/][0-9]{2}[-/][0-9]{2}T?", value)
        ):
            return pendulum.parse(value)
        raise ValueError("Could not parse `Date` from input.")

    def to_numpy(self):
        return np.datetime64(self.root)

    def __repr__(self):
        return self.root.__repr__()

    def __gt__(self, date1):
        return self.root > date1.root


class DateTime(RootModel):
    root: datetime.datetime

    @field_validator("root", mode="before")
    def validate_time(cls, value: Any) -> Any:
        if isinstance(value, str) and match(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T?[0-9]{2}:[0-9]{2}:?([0-9]{2})?Z?", value
        ):
            return pendulum.parse(value)
        raise ValueError("Could not parse `DateTime` from input.")

    def to_numpy(self):
        return np.datetime64(self.root)

    def __repr__(self):
        return self.root.__repr__()

    def __gt__(self, date1):
        return self.root > date1.root


class Time(RootModel):
    root: datetime.time

    @field_validator("root", mode="before")
    def validate_time(cls, value: Any) -> Any:
        if (
            isinstance(value, str)
            and len(value) >= 5
            and len(value) <= 9
            and match(r"[0-9]{2}:[0-9]{2}:?([0-9]{2})?Z?", value)
        ):
            return pendulum.parse(value).time()
        raise ValueError("Could not parse `Time` from input.")

    def to_numpy(self):
        raise NotImplementedError

    def __repr__(self):
        return self.time.__repr__()


class Year(RootModel):
    root: datetime.datetime

    @field_validator("root", mode="before")
    def validate_time(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) <= 4 and match(r"^\d{4}$", value):
            return pendulum.parse(value)
        raise ValueError("Could not parse `Year` from input.")

    def to_numpy(self):
        return np.datetime64(self.root)

    def __repr__(self):
        return self.root.__repr__()


class Duration(RootModel):
    root: datetime.timedelta

    @field_validator("root", mode="before")
    def validate_time(cls, value: Any) -> Any:
        if isinstance(value, str) and match(
            r"P[0-9]*Y?[0-9]*M?[0-9]*D?T?[0-9]*H?[0-9]*M?[0-9]*S?", value
        ):
            return pendulum.parse(value).as_timedelta()
        raise ValueError("Could not parse `Duration` from input.")

    def to_numpy(self):
        return np.timedelta64(self.root)

    def __repr__(self):
        return self.root.__repr__()


class TemporalInterval(RootModel):
    root: Annotated[
        list[Union[Year, Date, DateTime, Time, None]], Field(min_length=2, max_length=2)
    ]

    @field_validator("root")
    def validate_temporal_interval(cls, value: Any) -> Any:
        start = value[0]
        end = value[1]

        if start is None and end is None:
            raise ValueError("Could not parse `TemporalInterval` from input.")

        # Disambiguate the Time subtype
        if isinstance(start, Time) or isinstance(end, Time):
            if isinstance(start, Time) and isinstance(end, Time):
                raise ValueError(
                    "Ambiguous TemporalInterval, both start and end are of type `Time`"
                )
            if isinstance(start, Time):
                if end is None:
                    raise ValueError(
                        "Cannot disambiguate TemporalInterval, start is `Time` and end is `None`"
                    )
                logger.warning(
                    "Start time of temporal interval is of type `time`. Assuming same date as the end time."
                )
                start = DateTime(
                    root=pendulum.datetime(
                        end.root.year,
                        end.root.month,
                        end.root.day,
                        start.root.hour,
                        start.root.minute,
                        start.root.second,
                        start.root.microsecond,
                    ).to_rfc3339_string()
                )
            elif isinstance(end, Time):
                if start is None:
                    raise ValueError(
                        "Cannot disambiguate TemporalInterval, start is `None` and end is `Time`"
                    )
                logger.warning(
                    "End time of temporal interval is of type `time`. Assuming same date as the start time."
                )
                end = DateTime(
                    root=pendulum.datetime(
                        start.root.year,
                        start.root.month,
                        start.root.day,
                        end.root.hour,
                        end.root.minute,
                        end.root.second,
                        end.root.microsecond,
                    ).to_rfc3339_string()
                )

        if not (start is None or end is None) and start > end:
            raise ValueError("Start time > end time")

        return [start, end]

    @property
    def start(self):
        return self.root[0]

    @property
    def end(self):
        return self.root[1]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


class TemporalIntervals(RootModel):
    root: list[TemporalInterval]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item) -> TemporalInterval:
        return self.root[item]


GeoJson = Union[FeatureCollection, Feature, GeometryCollection, MultiPolygon, Polygon]
# The GeoJson spec (https://www.rfc-editor.org/rfc/rfc7946.html#ref-GJ2008) doesn't
# have a crs field anymore and recommends assuming it to be EPSG:4326, so we do the same.


class JobId(RootModel):
    root: str = Field(
        pattern=r"(eodc-jb-|jb-)[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}"
    )


ResultReference.model_rebuild()
ProcessNode.model_rebuild()
