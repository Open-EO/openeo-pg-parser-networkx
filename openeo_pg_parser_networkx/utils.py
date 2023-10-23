# This is forked from openeo-python-client, because I needed to have this unflatten subgraphs

from typing import Any

import pydantic

from openeo_pg_parser_networkx.pg_schema import ParameterReference, ResultReference


def parse_nested_parameter(parameter: Any):
    try:
        return ResultReference.parse_obj(parameter)
    except pydantic.error_wrappers.ValidationError:
        pass
    except TypeError:
        pass

    try:
        return ParameterReference.parse_obj(parameter)
    except pydantic.error_wrappers.ValidationError:
        pass
    except TypeError:
        pass

    return parameter


def find_result_node(flat_graph: dict) -> tuple[str, dict]:
    """
    Find result node in flat graph

    :return: tuple with node id (str) and node dictionary of the result node.
    """
    result_nodes = [
        (key, node) for (key, node) in flat_graph.items() if node.get("result")
    ]

    if len(result_nodes) == 1:
        return result_nodes[0]
    elif len(result_nodes) == 0:
        raise Exception("Found no result node in flat process graph")
    else:
        keys = [k for (k, n) in result_nodes]
        raise Exception(
            "Found multiple result nodes in flat process graph: {keys!r}".format(
                keys=keys
            )
        )


class ProcessGraphUnflattener:
    """
    Base class to process a flat graph representation of a process graph
    and unflatten it by resolving the "from_node" references.
    Subclassing and overriding certain methods allows to build a desired unflattened graph structure.
    """

    # Sentinel object for flagging a node "under construction" and detect graph cycles.
    _UNDER_CONSTRUCTION = object()

    def __init__(self, flat_graph: dict):
        self._flat_graph = flat_graph
        self._nodes = {}

    @classmethod
    def unflatten(cls, flat_graph: dict, **kwargs):
        """Class method helper to unflatten given flat process graph"""
        return cls(flat_graph=flat_graph, **kwargs).process()

    def process(self):
        """Process the flat process graph: unflatten it."""
        result_key, result_node = find_result_node(flat_graph=self._flat_graph)
        return self.get_node(result_key)

    def get_node(self, key: str) -> Any:
        """Get processed node by node key."""
        if key not in self._nodes:
            self._nodes[key] = self._UNDER_CONSTRUCTION
            node = self._process_node(self._flat_graph[key])
            self._nodes[key] = node
        elif self._nodes[key] is self._UNDER_CONSTRUCTION:
            raise Exception("Cycle in process graph")
        return self._nodes[key]

    def _process_node(self, node: dict) -> Any:
        """
        Overridable: generate process graph node from flat_graph data.
        """
        # Default implementation: basic validation/whitelisting, and only traverse arguments
        return dict(
            process_id=node["process_id"],
            arguments=self._process_value(value=node["arguments"]),
            **{k: node[k] for k in ["namespace", "description", "result"] if k in node},
        )

    def _process_from_node(self, key: str, node: dict) -> Any:
        """
        Overridable: generate a node from a flat_graph "from_node" reference
        """
        # Default/original implementation: keep "from_node" key and add resolved node under "node" key.
        # TODO: just return `self.get_node(key=key)`
        return {"from_node": key, "node": self.get_node(key=key)}

    def _process_from_parameter(self, name: str) -> Any:
        """
        Overridable: generate a node from a flat_graph "from_parameter" reference
        """
        # Default implementation:
        return {"from_parameter": name}

    def _process_child_graph(self, node_name: str, child_graph: dict):
        return {"process_graph": {node_name: child_graph}}

    def _resolve_from_node(self, key: str) -> dict:
        if key not in self._flat_graph:
            raise Exception(f"from_node reference {key!r} not found in process graph")
        return self._flat_graph[key]

    def _process_value(self, value) -> Any:
        if isinstance(value, dict):
            if "from_node" in value:
                key = value["from_node"]
                node = self._resolve_from_node(key=key)
                return self._process_from_node(key=key, node=node)
            elif "from_parameter" in value:
                name = value["from_parameter"]
                return self._process_from_parameter(name=name)
            elif "process_graph" in value:
                result_node_id, _ = find_result_node(value["process_graph"])
                return self._process_child_graph(
                    node_name=result_node_id,
                    child_graph=ProcessGraphUnflattener.unflatten(value["process_graph"]),
                )
            else:
                return {k: self._process_value(v) for (k, v) in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._process_value(v) for v in value]
        else:
            return value


import ast
from typing import Any

import numpy as np

from openeo_pg_parser_networkx.pg_schema import ParameterReference, ResultReference


def _format_nodes(pg, vars):
    nodes = []

    for var in vars:
        nodes.append(("variable", var, var, None))

    for node_id in pg:
        node = [n for n in pg.nodes if n[0] == node_id][0]

        # Special Cases
        # Array Element (indexing)
        if node[1]["process_id"] == "array_element":
            nodes.append(
                (
                    "array_element",
                    node[1]["node_name"],
                    node[1]["resolved_kwargs"]["data"],
                    node[1]["resolved_kwargs"]["index"],
                )
            )
            continue

        # Unary and Binary Operations

        formatted_node = (node[1]["process_id"], node[1]["node_name"])
        parameter_names = list(node[1]["resolved_kwargs"].keys())

        if len(parameter_names) == 1:
            parameter_names.append(None)

        for parameter in parameter_names:
            if parameter is None:
                formatted_node += (None,)
                continue
            parameter = node[1]["resolved_kwargs"][parameter]
            if isinstance(parameter, ParameterReference):
                formatted_node += (parameter.from_parameter,)
            elif isinstance(parameter, ResultReference):
                formatted_node += (parameter.from_node,)
            else:
                const_name = node[1]["node_name"] + "_const"
                nodes.append(
                    (
                        "const",
                        const_name,
                        parameter,
                        None,
                    )
                )
                formatted_node += (const_name,)

        nodes.append(formatted_node)
    return nodes


# Fit_Curve function builder

BIN_OP_MAPPING = {
    "multiply": ast.Mult(),
    "divide": ast.Div(),
    "subtract": ast.Sub(),
    "add": ast.Add(),
    "power": ast.Pow(),
    "mod": ast.Mod(),
}


def _generate_function_from_nodes(nodes: dict):
    temp_results = {}
    body = []

    for node_type, node_name, operand1, operand2 in nodes:
        # Value Operations
        if node_type == "array_element":
            value = ast.Subscript(
                value=ast.Name(id="parameters", ctx=ast.Load()),
                slice=ast.Index(value=ast.Num(n=int(operand2))),
                ctx=ast.Load(),
            )
        elif node_type == "const":
            value = ast.Num(n=operand1)
        elif node_type == "variable":
            value = ast.Name(id=operand1, ctx=ast.Load())

        # Binary Operations
        elif node_type in BIN_OP_MAPPING:
            value = ast.BinOp(
                left=temp_results[operand1],
                op=BIN_OP_MAPPING[node_type],
                right=temp_results[operand2],
            )

        # Unary Numpy Functions
        elif node_type in ["cos", "sin", "tan", "sqrt", "abs"]:
            value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr=node_type,
                    ctx=ast.Load(),
                ),
                args=[temp_results[operand1]],
                keywords=[],
            )

        # Binary Numpy Functions
        elif node_type in ["log"]:
            value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr=node_type,
                    ctx=ast.Load(),
                ),
                args=[temp_results[operand1], temp_results[operand2]],
                keywords=[],
            )

        result_var_store = ast.Name(id=node_name, ctx=ast.Store())
        result_var_load = ast.Name(id=node_name, ctx=ast.Load())
        temp_results[node_name] = result_var_load
        assign = ast.Assign(targets=[result_var_store], value=value)
        body.append(assign)

    return_stmt = ast.Return(value=temp_results[node_name])
    body.append(return_stmt)

    args = ast.arguments(
        posonlyargs=[],
        args=[
            ast.arg(arg="x", annotation=None),
            ast.arg(arg="parameters", annotation=None),
        ],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )

    func_ast = ast.FunctionDef(name="compute", args=args, body=body, decorator_list=[])

    numpy_import = ast.Import(names=[ast.alias(name="numpy", asname="np")])
    module = ast.Module(body=[numpy_import, func_ast], type_ignores=[])

    # Fix missing line numbers for the entire module
    module = ast.fix_missing_locations(module)

    code_obj = compile(module, "<string>", "exec")

    namespace = {}
    exec(code_obj, namespace)

    return namespace["compute"]


def generate_curve_fit_function(process_graph, variables=['x']):
    nodes = _format_nodes(process_graph, variables)
    return _generate_function_from_nodes(nodes)
