"""mena: run Mena machine-learned interatomic potentials from a command line, a local web page, or Python.

The command line and the web server never import torch themselves: every job runs in a worker process.
`calculator()`, `singlepoint()`, `relax()`, `run_md()`, and `available_models()` are lazy Python entry points.
"""

__version__ = "0.1.0"


def calculator(*args, **kwargs):
    """An ASE calculator for a Mena model; see mena_app.python_api.calculator."""
    from mena_app.python_api import calculator as _calculator

    return _calculator(*args, **kwargs)


def available_models(*args, **kwargs):
    """List available Mena models; see mena_app.python_api.available_models."""
    from mena_app.python_api import available_models as _available_models

    return _available_models(*args, **kwargs)


def singlepoint(*args, **kwargs):
    """Evaluate singlepoint energy, forces, and properties; see mena_app.python_api.singlepoint."""
    from mena_app.python_api import singlepoint as _singlepoint

    return _singlepoint(*args, **kwargs)


def relax(*args, **kwargs):
    """Relax atomic positions (and optionally cell); see mena_app.python_api.relax."""
    from mena_app.python_api import relax as _relax

    return _relax(*args, **kwargs)


def run_md(*args, **kwargs):
    """Run molecular dynamics; see mena_app.python_api.run_md."""
    from mena_app.python_api import run_md as _run_md

    return _run_md(*args, **kwargs)
