"""Weights download on first use (hash-checked, mirror fallback) and the per-user default model per structure kind."""

from __future__ import annotations

import hashlib
import json

import pytest

from mena_app import cli, models, settings, weights


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("MENA_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MENA_NO_DOWNLOAD", raising=False)
    monkeypatch.setattr(models.paths, "REPO_ROOT", tmp_path / "no-checkout")  # no weights/ folder to fall back on
    return tmp_path


def _fake_weights(tmp_path, data=b"mena-test-weights"):
    src = tmp_path / "src.pt"
    src.write_bytes(data)
    return src, hashlib.sha256(data).hexdigest()


def test_fresh_install_lists_shipped_models_as_downloadable(home):
    rows = {m["id"]: m for m in models.all_models()}
    assert set(rows) >= {"lam6-v1", "lam5-v1", "lam-400k"}
    for m in rows.values():
        if m.get("file") and m.get("release") == "preview":  # listed, never downloaded until it is published
            assert not m["available"] and not m["downloadable"]
        elif m.get("file"):
            assert not m["available"] and m["downloadable"]
            assert m["path"].startswith(str(home / "home" / "models"))
            assert len(m["urls"]) >= 2 and len(m["sha256"]) == 64
    assert models.choose("auto", "molecule")["id"] == "lam5-v1"
    assert models.choose("auto", "crystal")["id"] == "lam6-v1"
    assert models.choose("auto", "slab")["id"] == "lam6-v1"


def test_download_falls_back_to_next_mirror_and_checks_hash(home):
    src, digest = _fake_weights(home)
    bad = (home / "missing.pt").as_uri()
    m = {"id": "t", "title": "Test", "file": "t.pt", "sha256": digest, "urls": [bad, src.as_uri()], "size_mb": 0.0}
    path = weights.download(m, quiet=True)
    assert path.read_bytes() == src.read_bytes()
    assert weights.download(m, quiet=True) == path  # already there: no second download


def test_download_rejects_damaged_file_and_leaves_nothing(home):
    src, _ = _fake_weights(home)
    m = {"id": "t", "title": "Test", "file": "t.pt", "sha256": "0" * 64, "urls": [src.as_uri()]}
    with pytest.raises(weights.DownloadError, match="damaged"):
        weights.download(m, quiet=True)
    assert not any(weights.models_dir().iterdir())


def test_default_model_per_kind_drives_auto(home):
    assert models.choose("auto", "crystal")["id"] == "lam6-v1"
    models.set_default("crystal", "lam5-v1")
    m = models.choose("auto", "crystal")
    assert m["id"] == "lam5-v1" and "Your default" in m["choice_reason"]
    assert models.choose("auto", "slab")["id"] == "lam6-v1"  # other kinds keep their own default
    assert models.choose("lam-400k", "crystal")["id"] == "lam-400k"  # an explicit choice still wins
    models.set_default("crystal", "auto")
    assert settings.load()["default_model"] == {}
    assert models.choose("auto", "crystal")["id"] == "lam6-v1"


def test_default_model_rejects_unknown_or_wrong_domain(home, monkeypatch):
    with pytest.raises(models.ModelError):
        models.set_default("crystal", "no-such-model")
    with pytest.raises(models.ModelError):
        models.set_default("liquid", "lam5-v1")
    rows = [{"id": "molonly", "title": "Mol only", "domains": ["molecule"], "available": True}]
    monkeypatch.setattr(models, "all_models", lambda: rows)
    with pytest.raises(models.ModelError, match="does not cover crystals"):
        models.set_default("crystal", "molonly")


def test_unusable_saved_default_falls_back_with_a_note(home, monkeypatch):
    models.set_default("crystal", "lam5-v1")
    monkeypatch.setenv("MENA_NO_DOWNLOAD", "1")
    src, digest = _fake_weights(home)
    (home / "home" / "models.json").write_text(json.dumps({"models": [
        {"id": "lam-400k", "path": str(src), "backend": "lam", "domains": ["crystal"], "rank": {"crystal": 0}}]}), encoding="utf-8")
    m = models.choose("auto", "crystal")
    assert m["id"] == "lam-400k" and "not installed" in m["choice_reason"]


def test_cli_models_default_round_trip(home, capsys):
    assert cli.main(["models", "default", "crystal", "lam5-v1"]) == 0
    assert "lam5-v1" in capsys.readouterr().out
    assert settings.default_model("crystal") == "lam5-v1"
    assert cli.main(["models", "default", "crystal", "auto"]) == 0
    assert settings.default_model("crystal") is None


def test_entry_with_unsupported_backend_is_listed_but_never_chosen(home, capsys):
    src, _ = _fake_weights(home)
    (home / "home").mkdir(parents=True, exist_ok=True)
    (home / "home" / "models.json").write_text(json.dumps({"models": [
        {"id": "old", "path": str(src), "backend": "retired", "domains": ["molecule"], "rank": {"molecule": -1}}]}), encoding="utf-8")
    old = next(m for m in models.all_models() if m["id"] == "old")
    assert not old["available"] and "not supported" in old["unsupported"]
    assert models.choose("auto", "molecule")["id"] == "lam5-v1"
    with pytest.raises(models.ModelError, match="not supported"):
        models.choose("old", "molecule")
    assert cli.main(["models", "remove", "old"]) == 0
    assert all(m["id"] != "old" for m in models.all_models())
    assert cli.main(["models", "remove", "old"]) == 2
