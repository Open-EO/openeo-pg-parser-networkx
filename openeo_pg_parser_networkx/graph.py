from __future__ import annotations

import functools
import json
import logging
import random
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union
from uuid import UUID

import networkx as nx

from openeo_pg_parser_networkx.pg_schema import (
    ParameterReference,
    PGEdgeType,
    ProcessGraph,
    ProcessNode,
    ResultReference,
)
from openeo_pg_parser_networkx.utils import (
    ProcessGraphUnflattener,
    parse_nested_parameter,
)

logger = logging.getLogger(__name__)

ArgSubstitution = namedtuple("ArgSubstitution", ["arg_name", "access_func", "key"])


@dataclass
class EvalEnv:
    """
    Object to keep state of which node in the graph is currently being walked.
    """

    parent: Optional[EvalEnv]
    node: ProcessNode
    node_name: str
    process_graph_uid: str
    result: bool = False
    result_references_to_walk: list[EvalEnv] = field(default_factory=list)
    callbacks_to_walk: dict[str, ProcessGraph] = field(default_factory=dict)

    # This decorator makes this property not recompute each time it's called.
    @functools.cached_property
    def node_uid(self):
        return f"{self.node_name}-{self.process_graph_uid}"

    def __hash__(self) -> int:
        return hash(self.node_uid)

    def __repr__(self):
        return f"""\n
        ---------------------------------------
        EVAL_ENV {self.node_uid}
        parent: {self.parent}
        ---------------------------------------
        """


UNRESOLVED_CALLBACK_VALUE = "__UNRESOLVED_CALLBACK__"


class OpenEOProcessGraph:
    def __init__(self, pg_data: dict):
        self.G = nx.DiGraph()

        nested_raw_graph = self._unflatten_raw_process_graph(pg_data)
        self.nested_graph = self._parse_datamodel(nested_raw_graph)

        # Start parsing the graph at the result node of the top-level graph.
        self._EVAL_ENV = None

        self._parse_process_graph(self.nested_graph)

    @staticmethod
    def from_json(pg_json: str) -> OpenEOProcessGraph:
        return OpenEOProcessGraph(pg_data=json.loads(pg_json))

    @staticmethod
    def from_file(filepath: Union[str, Path]) -> OpenEOProcessGraph:
        return OpenEOProcessGraph(pg_data=json.load(open(filepath)))

    @staticmethod
    def _unflatten_raw_process_graph(raw_flat_graph: dict) -> dict:
        """
        Translates a flat process graph into a nested structure by resolving the from_node references.
        """
        if "process_graph" not in raw_flat_graph:
            raw_flat_graph = {"process_graph": raw_flat_graph}

        nested_graph = {
            "process_graph": {
                "root": ProcessGraphUnflattener.unflatten(raw_flat_graph["process_graph"])
            }
        }
        logger.warning("Deserialised process graph into nested structure")
        return nested_graph

    @staticmethod
    def _parse_datamodel(nested_graph: dict) -> ProcessGraph:
        """
        Parses a nested process graph into the Pydantic datamodel for ProcessGraph.
        """

        return ProcessGraph.parse_obj(nested_graph)

    def _parse_process_graph(self, process_graph: ProcessGraph, arg_name: str = None):
        """
        Start recursively walking a process graph from its result node and parse its information into self.G.
        This step passes process_graph.uid to make sure that each process graph operates within its own namespace so that nodes are unique.
        """

        for node_name, node in process_graph.process_graph.items():
            if node.result:
                self._EVAL_ENV = EvalEnv(
                    parent=self._EVAL_ENV,
                    node=node,
                    node_name=node_name,
                    process_graph_uid=process_graph.uid,
                    result=True,
                )
                if self._EVAL_ENV.parent:
                    self.G.add_edge(
                        self._EVAL_ENV.parent.node_uid,
                        self._EVAL_ENV.node_uid,
                        reference_type=PGEdgeType.Callback,
                        arg_name=arg_name,
                    )
                self._walk_node()
                self._EVAL_ENV = self._EVAL_ENV.parent
                return
        raise Exception("Process graph has no return node!")

    def _parse_argument(self, arg: any, arg_name: str, access_func: Callable):
        if isinstance(arg, ResultReference):
            # Finding a ResultReferences means that a new edge is required and the
            # node specified in `from_node` has to be added to the nodes that potentially need walking.
            from_node_eval_env = EvalEnv(
                parent=self._EVAL_ENV.parent,
                node=arg.node,
                node_name=arg.from_node,
                process_graph_uid=self._EVAL_ENV.process_graph_uid,
            )

            target_node = from_node_eval_env.node_uid

            self.G.add_edge(
                self._EVAL_ENV.node_uid,
                target_node,
                reference_type=PGEdgeType.ResultReference,
            )

            edge_data = self.G.edges[self._EVAL_ENV.node_uid, target_node]

            if "arg_substitutions" not in edge_data:
                edge_data["arg_substitutions"] = []

            edge_data["arg_substitutions"].append(
                ArgSubstitution(arg_name=arg_name, access_func=access_func, key=arg_name)
            )

            # Only add a subnode for walking if it's in the same process graph, otherwise you get infinite loops!
            if from_node_eval_env.process_graph_uid == self._EVAL_ENV.process_graph_uid:
                self._EVAL_ENV.result_references_to_walk.append(from_node_eval_env)

            access_func(new_value=arg, set_bool=True)

        # dicts and list parameters can contain further result or parameter references, so have to parse these exhaustively.
        elif isinstance(arg, dict):
            access_func(new_value={}, set_bool=True)

            for k, v in arg.items():
                access_func()[k] = None

                parsed_arg = parse_nested_parameter(v)

                # This access func business is necessary to let the program "remember" how to access and thus update this reference later
                sub_access_func = partial(
                    lambda key, access_func, new_value=None, set_bool=False: access_func()[
                        key
                    ]
                    if not set_bool
                    else access_func().__setitem__(key, new_value),
                    key=k,
                    access_func=access_func,
                )
                self._parse_argument(parsed_arg, arg_name, access_func=sub_access_func)

        elif isinstance(arg, list):
            access_func(new_value=[], set_bool=True)

            for i, element in enumerate(arg):
                access_func().append(None)
                parsed_arg = parse_nested_parameter(element)

                sub_access_func = partial(
                    lambda key, access_func, new_value=None, set_bool=False: access_func()[
                        key
                    ]
                    if not set_bool
                    else access_func().__setitem__(key, new_value),
                    key=i,
                    access_func=access_func,
                )
                self._parse_argument(parsed_arg, arg_name, access_func=sub_access_func)

        elif isinstance(arg, ProcessGraph):
            self._EVAL_ENV.callbacks_to_walk[arg_name] = arg

        else:
            access_func(new_value=arg, set_bool=True)

    def _walk_node(self):
        """
        Parse all the required information from the current node into self.G and recursively walk child nodes.
        """
        print(f"Walking node {self._EVAL_ENV.node_uid}")

        self.G.add_node(
            self._EVAL_ENV.node_uid,
            process_id=self._EVAL_ENV.node.process_id,
            resolved_kwargs={},
            node_name=self._EVAL_ENV.node_name,
            process_graph_uid=self._EVAL_ENV.process_graph_uid,
            result=self._EVAL_ENV.result,
        )

        for arg_name, unpacked_arg in self._EVAL_ENV.node.arguments.items():

            # Put the raw arg into the resolved_kwargs dict. If there are no further references within, that's already the right kwarg to pass on.
            # If there are further references, doing this will ensure that the container for these references is already there
            # and the access_functions can inject the resolved parameters later.
            self.G.nodes[self._EVAL_ENV.node_uid]["resolved_kwargs"][
                arg_name
            ] = unpacked_arg

            # This just points to the resolved_kwarg itself!
            access_func = partial(
                lambda node_uid, arg_name, new_value=None, set_bool=False: self.G.nodes[
                    node_uid
                ]["resolved_kwargs"][arg_name]
                if not set_bool
                else self.G.nodes[node_uid]["resolved_kwargs"].__setitem__(
                    arg_name, new_value
                ),
                node_uid=self._EVAL_ENV.node_uid,
                arg_name=arg_name,
            )
            self._parse_argument(unpacked_arg, arg_name, access_func=access_func)

        for arg_name, arg in self._EVAL_ENV.callbacks_to_walk.items():
            self.G.nodes[self._EVAL_ENV.node_uid]["resolved_kwargs"][
                arg_name
            ] = UNRESOLVED_CALLBACK_VALUE
            self._parse_process_graph(arg, arg_name=arg_name)

        for sub_eval_env in self._EVAL_ENV.result_references_to_walk:
            self._EVAL_ENV = sub_eval_env
            self._walk_node()

    def __iter__(self) -> str:
        """
        Traverse the process graph to yield nodes in the order they need to be executed.
        """
        top_level_graph = self._get_sub_graph(self.uid)
        visited_nodes = set()
        unlocked_nodes = [
            node for node, out_degree in top_level_graph.out_degree() if out_degree == 0
        ]
        while unlocked_nodes:
            node = unlocked_nodes.pop()
            visited_nodes.add(node)
            for child_node, _ in top_level_graph.in_edges(node):
                ready = True
                for _, uncle_node in top_level_graph.out_edges(child_node):
                    if uncle_node not in visited_nodes:
                        ready = False
                        break
                if ready and child_node not in visited_nodes:
                    unlocked_nodes.append(child_node)
            yield node

    def to_callable(
        self,
        process_registry: dict,
        results_cache: Optional[dict] = None,
        parameters: Optional[dict] = None,
    ) -> Callable:
        """
        Map the entire graph to a nested callable.
        """
        return self._map_node_to_callable(
            self.result_node, process_registry, results_cache, parameters
        )

    def _map_node_to_callable(
        self,
        node: str,
        process_registry: dict,
        results_cache: Optional[dict] = None,
        named_parameters: Optional[dict] = None,
    ) -> Callable:
        """Recursively walk the graph from a given node to construct a callable that calls the process
        implementations of the given node and all its parent nodes and passes intermediate results between
        them.
        """
        if results_cache is None:
            results_cache = {}

        if named_parameters is None:
            named_parameters = {}

        node_with_data = self.G.nodes(data=True)[node]
        process_impl = process_registry[node_with_data["process_id"]]

        static_parameters = node_with_data["resolved_kwargs"]
        parent_callables = []

        for _, source_node, data in self.G.out_edges(node, data=True):
            if data["reference_type"] == PGEdgeType.ResultReference:
                parent_callable = self._map_node_to_callable(
                    source_node,
                    process_registry=process_registry,
                    results_cache=results_cache,
                    named_parameters=named_parameters,
                )
                parent_callables.append(parent_callable)
            elif data["reference_type"] == PGEdgeType.Callback:
                callback = self._map_node_to_callable(
                    source_node,
                    process_registry=process_registry,
                    results_cache=results_cache,
                    named_parameters=named_parameters,
                )
                static_parameters[data["arg_name"]] = callback

        prebaked_process_impl = partial(
            process_impl, named_parameters=named_parameters, **static_parameters
        )

        def node_callable(*args, parent_callables, named_parameters=None, **kwargs):
            if parent_callables is None:
                parent_callables = []

            if named_parameters is None:
                named_parameters = {}

            # The node needs to first call all its parents, so that results are prepopulated in the results_cache
            for func in parent_callables:
                func(*args, named_parameters=named_parameters, **kwargs)

            try:
                # If this node has already been computed once, just grab that result from the results_cache instead of recomputing it.
                return results_cache.__getitem__(node)
            except KeyError:
                for _, source_node, data in self.G.out_edges(node, data=True):
                    if data["reference_type"] == PGEdgeType.ResultReference:
                        for arg_sub in data["arg_substitutions"]:
                            arg_sub.access_func(
                                new_value=results_cache[source_node], set_bool=True
                            )

                        kwargs[arg_sub.arg_name] = self.G.nodes(data=True)[node][
                            "resolved_kwargs"
                        ].__getitem__(arg_sub.arg_name)

                result = prebaked_process_impl(
                    *args, named_parameters=named_parameters, **kwargs
                )

                results_cache[node] = result

                return result

        return partial(node_callable, parent_callables=parent_callables)

    def _get_sub_graph(self, process_graph_id: str) -> nx.DiGraph:
        return self.G.subgraph(
            [
                node_id
                for node_id, data in self.G.nodes(data=True)
                if data["process_graph_uid"] == process_graph_id
            ]
        )

    @property
    def nodes(self) -> list:
        return list(self.G.nodes(data=True))

    @property
    def edges(self) -> list:
        return list(self.G.edges(data=True))

    @property
    def in_edges(self, node: str) -> list:
        return list(self.G.in_edges(node, data=True))

    @property
    def uid(self) -> UUID:
        return self.nested_graph.uid

    @property
    def required_processes(self) -> set[str]:
        """Return set of unique process_ids required to execute this process graph."""
        return {node[1] for node in self.G.nodes(data="process_id")}

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, OpenEOProcessGraph) and nx.is_isomorphic(self.G, __o.G)

    @property
    def result_node(self) -> str:
        return [
            node
            for node, in_degree in self.G.in_degree()
            if in_degree == 0
            if self.G.nodes(data=True)[node]["result"]
        ][0]

    def plot(self, reverse=False):
        if reverse:
            self.G = self.G.reverse()

        if self.G.number_of_nodes() < 1:
            logger.warning("Graph has no nodes, nothing to plot.")
            return

        sub_graphs = {
            process_graph_uid
            for _, process_graph_uid in nx.get_node_attributes(
                self.G, "process_graph_uid"
            ).items()
        }

        random.seed(42)
        node_colour_palette = {
            sub_graph_uid: random.randint(0, 255) for sub_graph_uid in sub_graphs
        }
        edge_colour_palette = {
            PGEdgeType.ResultReference: "blue",
            PGEdgeType.Callback: "red",
        }
        node_colours = [
            node_colour_palette[self.G.nodes(data=True)[node]["process_graph_uid"]]
            for node in self.G.nodes
        ]
        edge_colors = [
            edge_colour_palette.get(self.G.edges[edge]["reference_type"], "green")
            for edge in self.G.edges
        ]

        # To help with truncated labels
        import matplotlib.pyplot as plt

        plt.margins(x=0.2)

        nx.draw_circular(
            self.G,
            labels=nx.get_node_attributes(self.G, "node_name"),
            horizontalalignment="right",
            verticalalignment="top",
            node_color=node_colours,
            edge_color=edge_colors,
        )

        if reverse:
            self.G = self.G.reverse()
