import importlib.metadata

__version__ = importlib.metadata.version('openeo_pg_parser_networkx')

from openeo_pg_parser_networkx.graph import OpenEOProcessGraph
from openeo_pg_parser_networkx.process_registry import Process, ProcessRegistry
