# This is forked from openeo-python-client, because I needed to have this unflatten subgraphs

from typing import Any, Tuple


def find_result_node(flat_graph: dict) -> Tuple[str, dict]:
    """
    Find result node in flat graph

    :return: tuple with node id (str) and node dictionary of the result node.
    """
    result_nodes = [(key, node) for (key, node) in flat_graph.items() if node.get("result")]

    if len(result_nodes) == 1:
        return result_nodes[0]
    elif len(result_nodes) == 0:
        raise Exception("Found no result node in flat process graph")
    else:
        keys = [k for (k, n) in result_nodes]
        raise Exception(
            "Found multiple result nodes in flat process graph: {keys!r}".format(keys=keys))



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
            **{k: node[k] for k in ["namespace", "description", "result"] if k in node}
        )

    def _process_from_node(self, key: str, node: dict) -> Any:
        """
        Overridable: generate a node from a flat_graph "from_node" reference
        """
        # Default/original implementation: keep "from_node" key and add resolved node under "node" key.
        # TODO: just return `self.get_node(key=key)`
        return {
            "from_node": key,
            "node": self.get_node(key=key)
        }

    def _process_from_parameter(self, name: str) -> Any:
        """
        Overridable: generate a node from a flat_graph "from_parameter" reference
        """
        # Default implementation:
        return {"from_parameter": name}

    def _process_child_graph(self, child_graph: dict):
        return {"process_graph": child_graph}

    def _resolve_from_node(self, key: str) -> dict:
        if key not in self._flat_graph:
            raise Exception("from_node reference {k!r} not found in process graph".format(k=key))
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
                return self._process_child_graph(ProcessGraphUnflattener.unflatten(value["process_graph"]))
            else:
                return {k: self._process_value(v) for (k, v) in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._process_value(v) for v in value]
        else:
            return value
