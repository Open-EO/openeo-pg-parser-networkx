from __future__ import annotations

from eodc_pg_parser.process_implementations import *

available_functions = {
    load_collection,
    reduce_dimension,
    array_element,
    subtract,
    multiply,
    sum,
    divide,
    add,
    min,
    save_result,
    filter_spatial
}
process_registry = {func.__name__: func for func in available_functions}
