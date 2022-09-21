import xarray as xr


def load_collection(**kwargs):
    print(f"Ran load_collection with kwargs: {locals()}")
    return xr.open_dataset("../tests/data/boa.nc").to_array(dim='bands')

def reduce_dimension(**kwargs):
    kwargs["reducer"](**{'data': kwargs['data']})

def array_element(**kwargs):
    print(f"Ran array_element with kwargs: {locals()}")
    return kwargs["data"].sel(bands=kwargs["label"])

def divide(**kwargs):
    print(f"Ran divide with kwargs: {locals()}")
    return kwargs["x"] / kwargs["y"]

def subtract(**kwargs):
    print(f"Ran subtract with kwargs: {locals()}")
    return kwargs["x"] - kwargs["y"]

def multiply(**kwargs):
    print(f"Ran multiply with kwargs: {kwargs}")
    return kwargs["x"] * kwargs["y"]

def add(**kwargs):
    print(f"Ran add with kwargs: {locals()}")
    return kwargs["x"] + kwargs["y"]

def sum(**kwargs):
    print(f"Ran sum with kwargs: {locals()}")

    datas = [x for x in kwargs["data"] if isinstance(x, xr.DataArray)]

    final = datas[0]
    for data in datas[1:]:
        final = final + data

    return final

def min(**kwargs):
    print(f"Ran min with kwargs: {locals()}")
    return kwargs["data"].min()

def filter_spatial(**kwargs):
    print(f"Ran filter_spatial with kwargs: {locals()}")
    return "filter_spatial"

def save_result(**kwargs):
    print(f"Ran save_result with kwargs: {locals()}")
    return kwargs['data']
