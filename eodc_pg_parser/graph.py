import networkx as nx
import random
import logging
from openeo.internal.process_graph_visitor import ProcessGraphUnflattener
from eodc_pg_parser.pg_schema import ProcessNode, ResultReference, ParameterReference


logger = logging.getLogger(__name__)


class OpenEOProcessGraph(object):
    def __init__(self, pg_data):
        self.G = nx.MultiDiGraph()
        last_node = self._unflatten_process_graph(pg_data)
        self._walk_node(last_node, last_node.process_id)

    def _unflatten_process_graph(self, flat_graph: dict) -> ProcessNode:
        pg_unflattener = ProcessGraphUnflattener(flat_graph)
        graph = pg_unflattener.unflatten(flat_graph["process_graph"])
        last_node = ProcessNode.parse_obj(graph)
        logger.warning("Deserialised process graph into nested structure")
        return last_node

    def _walk_node(self, node: ProcessNode, node_id: str):
        # 1. Find the connected nodes. These can either be ResultReferences or ParameterReferences 
        # (or UDFs I guess)

        self.G.add_node(node_id, node=node, resolved_kwargs={})

        # Walk any dependencies first!
        for arg_name, arg_container in node.arguments.items():
            arg = arg_container.__root__
            # Create edges for result references
            if isinstance(arg, ResultReference):
                self._walk_node(arg.node, node_id=arg.from_node)
                self.G.add_edge(node_id, arg.from_node, reference_type=arg, arg_name=arg_name)

            # Parameter references need to be resolved from the dependant nodes upwards.
            if isinstance(arg, ParameterReference):
                pass

            # Construct the argument list
            else:
                self.G.nodes[node_id]["resolved_kwargs"][arg_name] = arg
    
        # TODO: Handle reducers


    def plot(self):
        if self.G.number_of_nodes() < 1:
            logger.warning("Graph has no nodes, nothing to plot.")
            return

        n_colours = (
            max(nx.shortest_path_length(self.G, source=self.get_root_node()).values())
            + 1
        )
        colour_palette = [random.randint(0, 255) for _ in range(n_colours)]
        colours = [colour_palette[self.get_node_depth(node)] for node in self.G.nodes]

        nx.draw_kamada_kawai(
            self.G,
            labels={node: node for node in self.G.nodes},
            horizontalalignment="right",
            verticalalignment="top",
            node_color=colours,
        )
        nx.draw_networkx_edge_labels(
            self.G,
            nx.kamada_kawai_layout(self.G),
            nx.get_edge_attributes(self.G, "name"),
            font_color="#00211e",
        )

    def get_root_node(self):
        return next(nx.topological_sort(self.G))

    def get_node_depth(self, node):
        return nx.shortest_path_length(self.G, source=self.get_root_node(), target=node)

