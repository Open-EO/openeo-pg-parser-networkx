def load_collection(**kwargs):
    print(f"Ran load_collection with kwargs: {locals()}")
    return "load_collection"


def reduce_dimension(**kwargs):
    print(f"Ran reduce_dimension with kwargs: {locals()}")
    kwargs["reducer"]()
    return "reduce_dimension"


def array_element(**kwargs):
    print(f"Ran array_element with kwargs: {locals()}")
    return "array_element"


def subtract(**kwargs):
    print(f"Ran subtract with kwargs: {locals()}")
    return "subtract"


def multiply(**kwargs):
    print(f"Ran multiply with kwargs: {locals()}")
    return "multiply"


def sum(**kwargs):
    print(f"Ran sum with kwargs: {locals()}")
    return "sum"


def divide(**kwargs):
    print(f"Ran divide with kwargs: {locals()}")
    return "divide"


def min(**kwargs):
    print(f"Ran min with kwargs: {locals()}")
    return "min"


def save_result(**kwargs):
    print(f"Ran save_result with kwargs: {locals()}")
    return "save_result"
