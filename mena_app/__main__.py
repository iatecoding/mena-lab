"""Allow `python -m mena_app ...` alongside the installed `mena` script."""

from __future__ import annotations

import sys

from mena_app.cli import main

if __name__ == "__main__":
    sys.exit(main())
