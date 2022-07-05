from eodc_pg_parser.models import ProcessGraph, ProcessNode

from unittest import TestCase


class PGTester(TestCase):
    def test_json_parsing(self):
        pg = ProcessGraph.parse_file("tests/data/fit_rf_pg_0.json")
        print(pg)
            
