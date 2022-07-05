import networkx as nx
import random
import logging

logger = logging.getLogger(__name__)


class OpenEOGraph(object):
    def __init__(self, pg_data):
        self.G = nx.MultiDiGraph()
        self._translate_process_graph(pg_data)

    def _translate_process_graph(self, pg_data_json):
        self._walk_process_graph(pg_data_json)

    def _walk_process_graph(self, parent):
        for child, value in parent.items():
            if isinstance(value, dict):
                if "process_id" in value.keys():
                    sub_nodes = self._parse_process_node(child, value)
                    for child, value in parent.items():
                        self._walk_process_graph(sub_nodes)

        # Expand all arrays fields into the node attributes

    def _parse_process_node(self, node_name, node_data):
        self.G.add_node(process_id=node_data["process_id"])

        args = node_data.get("arguments", {})
        from_node = (
            node_data.get("arguments", {}).get("data", {}).get("from_parameter", None)
        )
        if from_node:
            self.G.add_edge(from_node, node_name)

        sub_graphs = args.get("reducer", {}).get("process_graph", {}).values()

        return sub_graphs

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


# def cytoscape_graph(selfdata, attrs=None, name="name", ident="id"):
#     multigraph = data.get("multigraph")
#     directed = data.get("directed")
#     if multigraph:
#         graph = nx.MultiGraph()
#     else:
#         graph = nx.Graph()
#     if directed:
#         graph = graph.to_directed()
#     graph.graph = dict(data.get("data"))
#     for d in data["elements"]["nodes"]:
#         node_data = d["data"].copy()
#         node = d["data"]["value"]

#         if d["data"].get(name):
#             node_data[name] = d["data"].get(name)
#         if d["data"].get(ident):
#             node_data[ident] = d["data"].get(ident)

#         graph.add_node(node)
#         graph.nodes[node].update(node_data)

#     for d in data["elements"]["edges"]:
#         edge_data = d["data"].copy()
#         sour = d["data"]["source"]
#         targ = d["data"]["target"]
#         if multigraph:
#             key = d["data"].get("key", 0)
#             graph.add_edge(sour, targ, key=key)
#             graph.edges[sour, targ, key].update(edge_data)
#         else:
#             graph.add_edge(sour, targ)
#             graph.edges[sour, targ].update(edge_data)
#     return graph
