import importlib
from typing import List


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_names(path: str, module_name: str, names: List[str]):
    module = import_file(path, module_name)
    result = []
    for name in names:
        if hasattr(module, name):
            result.append(getattr(module, name))
        else:
            raise AttributeError(f'No such name {name} in module {module}.')
    return result
