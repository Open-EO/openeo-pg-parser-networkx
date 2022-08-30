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
    min,
    save_result,
}
process_registry = {func.__name__: func for func in available_functions}
