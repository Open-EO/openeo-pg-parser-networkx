from eodc_pg_parser.pg_schema import ProcessNode
from openeo.internal.process_graph_visitor import ProcessGraphVisitor, ProcessGraphUnflattener

from unittest import TestCase
import json

class PGTester(TestCase):
    def test_graph_visitor(self):
        pg_1 = json.load(open("tests/data/fit_rf_pg_0.json", mode="r"))

        pg_visitor = ProcessGraphVisitor()
        graph = pg_visitor.accept_process_graph(pg_1["process_graph"])

    def test_ProcessGraphUnflattener(self):
        flat_graph = json.load(open("tests/data/fit_rf_pg_0.json", mode="r"))

        pg_unflattener = ProcessGraphUnflattener(flat_graph)
        graph = pg_unflattener.unflatten(flat_graph["process_graph"])
        print(graph)

    def test_unflattened_data_model(self):
        flat_graph = json.load(open("tests/data/fit_rf_pg_0.json", mode="r"))
        pg_unflattener = ProcessGraphUnflattener(flat_graph)
        graph = pg_unflattener.unflatten(flat_graph["process_graph"])
        parsed_graph = ProcessNode.parse_obj(graph)
        k = [arg for arg in parsed_graph.arguments]
        print("")