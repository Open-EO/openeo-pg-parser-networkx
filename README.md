# PG Parser

## Installation
`poetry install` to setup the python venv.
Use that venv as the kernel for the ipynb notebooks.
`poetry run python -m pytest` to run the test suite.
`poetry add some_new_dependency` to add a new dependency.

## Pydantic Typing

Pydantic datamodels were downloaded using:
`poetry run datamodel-codegen --url https://openeo.org/documentation/1.0/developers/api/assets/pg-schema.json --output eodc_pg_parser/models/pg_schema.py --input-file-type jsonschema`




