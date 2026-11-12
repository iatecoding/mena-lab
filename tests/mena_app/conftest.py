"""Tests run against a private, empty MENA_HOME (never the developer's ~/.mena) and never download model weights: a
model counts as installed only if its file is on disk (a source checkout's weights/ folder).
MENA_TEST_DOWNLOAD=1 downloads the published weights once into that private home first, so the end-to-end tests run
(CI does this on every push)."""

import os
import tempfile

if not os.environ.get("MENA_TEST_DOWNLOAD"):
    os.environ["MENA_NO_DOWNLOAD"] = "1"
os.environ["MENA_HOME"] = tempfile.mkdtemp(prefix="mena-test-home-")

if os.environ.get("MENA_TEST_DOWNLOAD"):
    from mena_app import models

    for _m in models.all_models():
        if _m.get("downloadable") and not _m.get("available"):
            models.ensure_file(_m)
