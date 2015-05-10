class Material(type):
    param_names = []

def check_params_valid(typ, params):
    return set(typ.param_names) == set(params.keys())

