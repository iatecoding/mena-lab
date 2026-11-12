"""Model weight files: where they are kept and how they are downloaded. Standard library only.

A shipped model entry names a `file`, its `sha256` and a list of `urls` (mirrors, tried in order). Downloads go to
~/.mena/models/<file> (or $MENA_HOME/models). The hash is checked after every download, so a broken or tampered file
is never used; it is an integrity check, not something the user has to look at.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from mena_app import paths


class DownloadError(RuntimeError):
    pass


RETRIES = 3  # per mirror, for dropped connections


def models_dir() -> Path:
    d = paths.data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def local_path(model: dict) -> Path | None:
    return models_dir() / model["file"] if model.get("file") else None


def downloadable(model: dict) -> bool:
    return bool(model.get("file") and model.get("urls") and model.get("sha256"))


def auto_download_allowed() -> bool:
    return os.environ.get("MENA_NO_DOWNLOAD", "").strip().lower() not in ("1", "true", "yes")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, dest: Path, label: str, quiet: bool) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "mena-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if not quiet and total:
                pct = int(100 * done / total)
                if pct != last:
                    print(f"\r  {label}: {done / 1e6:5.1f} / {total / 1e6:.1f} MB ({pct}%)", end="", file=sys.stderr, flush=True)
                    last = pct
    if not quiet:
        print(file=sys.stderr, flush=True)


def download(model: dict, quiet: bool = False) -> Path:
    """Download the model's file from the first mirror that works and verify its hash. Returns the local path."""
    if not downloadable(model):
        raise DownloadError(f"{model.get('title', model['id'])} has no download source; add the file with `mena models add`.")
    dest = local_path(model)
    if dest.is_file() and sha256(dest) == model["sha256"]:
        return dest
    label = f"{model.get('title', model['id'])} ({model.get('size_mb', '?')} MB)"
    if not quiet:
        print(f"Downloading {label} to {dest.parent}", file=sys.stderr, flush=True)
    errors = []
    for url in model["urls"]:
        for attempt in range(RETRIES):
            fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
            os.close(fd)
            tmp_p = Path(tmp)
            try:
                _fetch(url, tmp_p, model["id"], quiet)
                got = sha256(tmp_p)
                if got != model["sha256"]:
                    raise DownloadError(f"file from {url} is damaged (hash mismatch)")
                tmp_p.replace(dest)
                return dest
            except Exception as exc:  # noqa: BLE001 - retry a dropped connection, else try the next mirror
                errors.append(f"{url}: {exc}")
                missing = isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404)
                if missing or attempt == RETRIES - 1:
                    break
                time.sleep(2 ** attempt)
            finally:
                if tmp_p.exists():
                    tmp_p.unlink()
    raise DownloadError(f"could not download {model['id']}:\n  " + "\n  ".join(errors))
