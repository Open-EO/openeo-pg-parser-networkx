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


def format_nodes(pg, vars):
    nodes = []

    for var in vars:
        nodes.append(("variable", var, var, None))

    for node_id in pg:
        node = [n for n in pg.nodes if n[0] == node_id][0]

        formatted_node = (node[1]["process_id"], node[1]["node_name"])

        # Parameters

        parameter_names = list(node[1]["resolved_kwargs"].keys())

        if parameter_names[0] == "data":
            formatted_node += (node[1]["resolved_kwargs"]["index"],)
            formatted_node += (None,)
            nodes.append(formatted_node)
            continue

        if "x" in parameter_names:
            if isinstance(node[1]["resolved_kwargs"]["x"], ParameterReference):
                formatted_node += (node[1]["resolved_kwargs"]["x"].from_parameter,)
            elif isinstance(node[1]["resolved_kwargs"]["x"], ResultReference):
                formatted_node += (node[1]["resolved_kwargs"]["x"].from_node,)
            else:
                const_name = node[1]["node_name"] + "_x"
                nodes.append(
                    (
                        "const",
                        const_name,
                        node[1]["resolved_kwargs"]["x"],
                        None,
                    )
                )
                formatted_node += (const_name,)
        else:
            formatted_node += (None,)
        if "y" in parameter_names:
            if isinstance(node[1]["resolved_kwargs"]["y"], ParameterReference):
                formatted_node += (node[1]["resolved_kwargs"]["y"].from_parameter,)
            elif isinstance(node[1]["resolved_kwargs"]["y"], ResultReference):
                formatted_node += (node[1]["resolved_kwargs"]["y"].from_node,)
            else:
                const_name = node[1]["node_name"] + "_y"
                nodes.append(
                    (
                        "const",
                        const_name,
                        node[1]["resolved_kwargs"]["y"],
                        None,
                    )
                )
                formatted_node += (const_name,)
        else:
            formatted_node += (None,)

        nodes.append(formatted_node)
    return nodes


# Fit_Curve function builder


def generate_function_from_nodes(nodes):
    temp_results = {}
    body = []

    for node_type, node_name, operand1, operand2 in nodes:
        if node_type == "array_element":
            value = ast.Subscript(
                value=ast.Name(id="parameter", ctx=ast.Load()),
                slice=ast.Index(value=ast.Num(n=int(operand1))),
                ctx=ast.Load(),
            )
        elif node_type == "const":
            value = ast.Num(n=operand1)
        elif node_type == "variable":
            value = ast.Name(id=operand1, ctx=ast.Load())
        elif node_type == "multiply":
            value = ast.BinOp(
                left=temp_results[operand1], op=ast.Mult(), right=temp_results[operand2]
            )
        elif node_type == "divide":
            value = ast.BinOp(
                left=temp_results[operand1], op=ast.Div(), right=temp_results[operand2]
            )
        elif node_type == "subtract":
            value = ast.BinOp(
                left=temp_results[operand1], op=ast.Sub(), right=temp_results[operand2]
            )
        elif node_type == "add":
            value = ast.BinOp(
                left=temp_results[operand1], op=ast.Add(), right=temp_results[operand2]
            )

        elif node_type == "cos":
            value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()), attr="cos", ctx=ast.Load()
                ),
                args=[temp_results[operand1]],
                keywords=[],
            )
        elif node_type == "sin":
            value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()), attr="sin", ctx=ast.Load()
                ),
                args=[temp_results[operand1]],
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
