"""One module per task; each has run(ctx) -> report (see mena_app/report.py). Imported only inside the worker process."""

from importlib import import_module

TASKS = ("singlepoint", "relax", "md", "vibrations", "phonons", "neb", "benchmark")


def load(task: str):
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    return import_module(f"mena_app.tasks.{task}")
