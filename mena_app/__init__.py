"""mena: run Mena machine-learned interatomic potentials from a command line, a local web page, or Python.

The command line and the web server never import torch themselves: every job runs in a worker process.
`calculator()` is the Python entry point; it imports torch when called.
"""

__version__ = "0.1.0"


def calculator(*args, **kwargs):
    """An ASE calculator for a Mena model; see mena_app.python_api.calculator."""
    from mena_app.python_api import calculator as _calculator

    return _calculator(*args, **kwargs)
