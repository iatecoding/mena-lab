"""Opt-in step-level dipole & polarizability capture helpers for `MDEngine`."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from lam4.calc.precision_hygiene import restore_charged_dipole_f64
from lam4.calc.spectroscopy import E_ANG_TO_DEBYE, polarizability_from_sums

_DIPOLE_PROPS = frozenset({"dipole", "dipole_debye", "dipole_norm_debye"})
_POL_PROPS = frozenset({"polarizability", "polarizability_iso", "polarizability_aniso", "alpha"})
_PASSTHROUGH_PROPS = frozenset({"energy", "forces", "stress", "stress_voigt", "positions"})
_VALID_RECORD_PROPS = _DIPOLE_PROPS | _POL_PROPS | _PASSTHROUGH_PROPS


def init_md_record_properties(eng: Any, record_properties: Sequence[str]) -> None:
    """Initialize property-recording attributes on `MDEngine` with zero device allocation when disabled."""
    eng.record_properties = ()
    eng._want_dipole = False
    eng._want_pol = False
    eng._want_props = False
    eng._has_aniso = bool(getattr(eng.model, "polarization", None) is not None and getattr(eng.model.polarization, "aniso", False))
    eng._prop_history = {}
    configure_md_record_properties(eng, record_properties)


def configure_md_record_properties(eng: Any, record_properties: Sequence[str]) -> None:
    """Validate `record_properties` and lazily allocate device buffers only when opted in."""
    raw = (record_properties,) if isinstance(record_properties, str) else tuple(record_properties)
    seen: list[str] = []
    for p in raw:
        key = str(p).strip()
        if key not in _VALID_RECORD_PROPS:
            raise ValueError(f"Unsupported record_properties entry {p!r}; expected one of {sorted(_VALID_RECORD_PROPS)}")
        if key not in seen:
            seen.append(key)
    eng.record_properties = tuple(seen)
    want_dip = any(p in _DIPOLE_PROPS for p in eng.record_properties)
    want_pol = any(p in _POL_PROPS for p in eng.record_properties)
    if want_dip == eng._want_dipole and want_pol == eng._want_pol:
        return
    fkw = dict(dtype=eng.state_dt, device=eng.dev)
    if want_dip and not hasattr(eng, "dip"):
        eng.dip = torch.zeros(3, **fkw)
        eng._dip_cur = torch.zeros(3, **fkw)
        eng.dip_win = torch.zeros((eng.window, 3), **fkw)
    if want_pol and not hasattr(eng, "alpha_iso"):
        eng.alpha_iso = torch.zeros((), **fkw)
        eng._ai_cur = torch.zeros((), **fkw)
        eng.alpha_iso_win = torch.zeros(eng.window, **fkw)
        eng.alpha_aniso = torch.zeros(5, **fkw)
        eng._aa_cur = torch.zeros(5, **fkw)
        eng.alpha_aniso_win = torch.zeros((eng.window, 5), **fkw)
    eng._want_dipole = want_dip
    eng._want_pol = want_pol
    eng._want_props = bool(want_dip or want_pol)
    if eng.graph is not None:
        eng.graph = None


def capture_step_properties(eng: Any, out: dict[str, Any]) -> None:
    """Copy step-level dipole and polarizability sums into preallocated static device scalars."""
    sdt = eng.state_dt
    if eng._want_dipole:
        dip_t = out.get("dipole")
        if dip_t is not None:
            eng._dip_cur.copy_(dip_t[0].detach().to(sdt))
        else:
            eng._dip_cur.zero_()
    if eng._want_pol:
        ai_t = out.get("alpha")
        if ai_t is not None:
            eng._ai_cur.copy_(ai_t[: eng.n].detach().to(sdt).sum())
            aa_t = out.get("alpha_aniso")
            if aa_t is not None:
                eng._aa_cur.copy_(aa_t[: eng.n].detach().to(sdt).sum(dim=0))
            else:
                eng._aa_cur.zero_()
        else:
            eng._ai_cur.zero_()
            eng._aa_cur.zero_()


def record_win_step_properties(eng: Any, slot: int) -> None:
    if eng._want_dipole:
        eng.dip_win[slot].copy_(eng._dip_cur)
    if eng._want_pol:
        eng.alpha_iso_win[slot].copy_(eng._ai_cur)
        eng.alpha_aniso_win[slot].copy_(eng._aa_cur)


def flush_step_properties(eng: Any) -> None:
    if eng._want_dipole:
        eng.dip.copy_(eng._dip_cur)
    if eng._want_pol:
        eng.alpha_iso.copy_(eng._ai_cur)
        eng.alpha_aniso.copy_(eng._aa_cur)


def extra_state_tensors(eng: Any) -> tuple[Any, ...]:
    if not eng._want_props:
        return ()
    extra: list[Any] = []
    if eng._want_dipole:
        extra.append(eng.dip)
    if eng._want_pol:
        extra.extend((eng.alpha_iso, eng.alpha_aniso))
    return tuple(extra)


def collect_window_properties(eng: Any) -> dict[str, Any]:
    """Transfer one window's recorded device property ring buffers in ONE D2H copy and append history."""
    parts: list[torch.Tensor] = []
    if eng._want_dipole:
        parts.append(eng.dip_win.reshape(-1))
    if eng._want_pol:
        parts.append(eng.alpha_iso_win.reshape(-1))
        parts.append(eng.alpha_aniso_win.reshape(-1))
    buf = torch.cat(parts, dim=0).cpu().to(torch.float64).numpy()
    cur, w = 0, eng.window
    out: dict[str, Any] = {}
    step_end = int(eng.stats["steps"])
    eng._prop_history.setdefault("step", []).append(np.arange(step_end - w + 1, step_end + 1, dtype=np.int64))
    if eng._want_dipole:
        dip_w = buf[cur : cur + 3 * w].reshape(w, 3).copy()
        cur += 3 * w
        origin = getattr(eng, "_origin_f64", None)
        if origin is not None and eng.qv != 0.0 and eng.model.trunk.electro(eng.src.domain, eng.pbc_any, bool(getattr(eng, "periodic_electro", False))):
            dip_w = dip_w + float(eng.qv) * origin[None, :]
        dip_deb_w = dip_w * E_ANG_TO_DEBYE
        eng._prop_history.setdefault("dipole", []).append(dip_w)
        eng._prop_history.setdefault("dipole_debye", []).append(dip_deb_w)
        out["dipole"] = dip_w[-1].copy()
        out["dipole_debye"] = dip_deb_w[-1].copy()
        out["dipole_norm_debye"] = float(np.linalg.norm(dip_deb_w[-1]))
        out["step_dipoles"] = dip_w
        out["step_dipoles_debye"] = dip_deb_w
    if eng._want_pol:
        ai_w = buf[cur : cur + w]
        cur += w
        aa_w = buf[cur : cur + 5 * w].reshape(w, 5)
        cur += 5 * w
        pol_w = np.empty((w, 3, 3), dtype=np.float64)
        iso_w = np.empty(w, dtype=np.float64)
        aniso_w = np.empty(w, dtype=np.float64)
        use_aniso = eng._has_aniso and ((not eng.pbc_any) or bool(getattr(eng, "periodic_electro", False)))
        for k in range(w):
            pol_w[k], iso_w[k], aniso_w[k] = polarizability_from_sums(
                float(ai_w[k]), aa_w[k] if use_aniso else None,
            )
        eng._prop_history.setdefault("polarizability", []).append(pol_w)
        eng._prop_history.setdefault("polarizability_iso", []).append(iso_w)
        eng._prop_history.setdefault("polarizability_aniso", []).append(aniso_w)
        out["polarizability"] = pol_w[-1].copy()
        out["polarizability_iso"] = float(iso_w[-1])
        out["polarizability_aniso"] = float(aniso_w[-1])
        out["step_polarizabilities"] = pol_w
        out["step_polarizabilities_iso"] = iso_w
        out["step_polarizabilities_aniso"] = aniso_w
    return out


def append_recorded_frame(eng: Any, prop_w: dict[str, Any] | None) -> None:
    s33 = np.array(eng.s.cpu().numpy(), copy=True)
    frame: dict[str, Any] = dict(
        step=eng.stats["steps"], positions=eng.positions().copy(), energy=eng.energy_ev(),
        forces=np.array(eng.f.cpu().numpy(), copy=True), stress=s33, stress_voigt=s33[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]],
    )
    if prop_w is not None:
        frame.update(prop_w)
    eng.frames.append(frame)


def current_dipole(eng: Any) -> np.ndarray:
    if not eng._want_dipole:
        raise RuntimeError("Dipole recording not enabled; pass record_properties=('dipole', ...) to MDEngine or run().")
    dip = eng.dip.detach().cpu().to(torch.float64).numpy()
    origin = getattr(eng, "_origin_f64", None)
    has_dip = bool(origin is not None and eng.model.trunk.electro(eng.src.domain, eng.pbc_any, bool(getattr(eng, "periodic_electro", False))))
    return restore_charged_dipole_f64(dip, float(getattr(eng, "qv", 0.0)), origin if origin is not None else np.zeros(3), has_dip)


def current_dipole_debye(eng: Any) -> np.ndarray:
    return current_dipole(eng) * E_ANG_TO_DEBYE


def current_pol_tuple(eng: Any) -> tuple[np.ndarray, float, float]:
    if not eng._want_pol:
        raise RuntimeError("Polarizability recording not enabled; pass record_properties=('polarizability', ...) to MDEngine or run().")
    use_aniso = eng._has_aniso and ((not eng.pbc_any) or bool(getattr(eng, "periodic_electro", False)))
    buf = torch.cat([eng.alpha_iso.view(1), eng.alpha_aniso.view(5)]).detach().cpu().to(torch.float64).numpy()
    return polarizability_from_sums(float(buf[0]), buf[1:6] if use_aniso else None)


def validate_record_props(record_properties: Sequence[str]) -> tuple[str, ...]:
    """Validate and deduplicate `record_properties` entries."""
    raw = (record_properties,) if isinstance(record_properties, str) else tuple(record_properties)
    seen: list[str] = []
    for p in raw:
        key = str(p).strip()
        if key not in _VALID_RECORD_PROPS:
            raise ValueError(f"Unsupported record_properties entry {p!r}; expected one of {sorted(_VALID_RECORD_PROPS)}")
        if key not in seen:
            seen.append(key)
    return tuple(seen)


def finalize_engine_props(eng: Any, rec_props: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    """Collect requested step-level & passthrough properties from a finished `MDEngine`."""
    if not rec_props:
        return None
    out: dict[str, np.ndarray] = dict(eng.trajectory_properties()) if eng._want_props else {}
    if "dipole_norm_debye" in rec_props and "dipole_debye" in out:
        out["dipole_norm_debye"] = np.linalg.norm(out["dipole_debye"], axis=-1)
    if any(p in _PASSTHROUGH_PROPS for p in rec_props) and eng.frames:
        out.setdefault("step", np.asarray([int(fr["step"]) for fr in eng.frames], dtype=np.int64))
        for p in rec_props:
            if p in _PASSTHROUGH_PROPS:
                out[p] = np.asarray([fr[p] for fr in eng.frames], dtype=np.float64)
    return out


def needs_passthrough_frames(rec_props: tuple[str, ...]) -> bool:
    return any(p in _PASSTHROUGH_PROPS for p in rec_props)


def ase_collect_step(hist: dict[str, list[Any]], step: int, work: Any, base_calc: Any, rec_props: tuple[str, ...]) -> None:
    """Append step-level `record_properties` during an ASE MD fallback run."""
    hist.setdefault("step", []).append(step)
    res = getattr(base_calc, "results", {}) or {}
    if any(p in _DIPOLE_PROPS for p in rec_props):
        dip = np.asarray(res.get("dipole", np.zeros(3)), dtype=np.float64).copy()
        dip_deb = np.asarray(res.get("dipole_debye", dip * E_ANG_TO_DEBYE), dtype=np.float64).copy()
        hist.setdefault("dipole", []).append(dip)
        hist.setdefault("dipole_debye", []).append(dip_deb)
        if "dipole_norm_debye" in rec_props:
            hist.setdefault("dipole_norm_debye", []).append(float(np.linalg.norm(dip_deb)))
    if any(p in _POL_PROPS for p in rec_props):
        hist.setdefault("polarizability", []).append(np.asarray(res.get("polarizability", np.zeros((3, 3))), dtype=np.float64).copy())
        hist.setdefault("polarizability_iso", []).append(float(res.get("polarizability_iso", 0.0)))
        hist.setdefault("polarizability_aniso", []).append(float(res.get("polarizability_aniso", 0.0)))
    for p in rec_props:
        if p == "positions":
            hist.setdefault("positions", []).append(np.asarray(work.get_positions(), dtype=np.float64).copy())
        elif p == "energy":
            e_v = work.calc.results["energy"] if "energy" in work.calc.results else work.get_potential_energy()
            hist.setdefault("energy", []).append(float(e_v))
        elif p == "forces":
            f_v = work.calc.results["forces"] if "forces" in work.calc.results else work.get_forces()
            hist.setdefault("forces", []).append(np.asarray(f_v, dtype=np.float64).copy())
        elif p in ("stress", "stress_voigt"):
            s_v = np.asarray(res.get("stress", np.zeros(6)), dtype=np.float64).copy()
            hist.setdefault(p, []).append(s_v if p == "stress_voigt" else np.array(
                [[s_v[0], s_v[5], s_v[4]], [s_v[5], s_v[1], s_v[3]], [s_v[4], s_v[3], s_v[2]]], dtype=np.float64,
            ))


class TrajSink:
    """Write ASE `.traj`, `.extxyz`/`.xyz`, or custom trajectory writers with cached E/F/stress."""

    def __init__(self, target: Any | None) -> None:
        import os
        from pathlib import Path

        self._target, self._path, self._traj, self._owns, self.count = target, None, None, False, 0
        if isinstance(target, (str, os.PathLike)):
            self._path = Path(target)
            if self._path.suffix.lower() == ".traj":
                from ase.io.trajectory import Trajectory

                self._traj, self._owns = Trajectory(str(self._path), "w"), True

    def write(self, atoms: Any, *, energy: float | None = None, forces: np.ndarray | None = None, stress: np.ndarray | None = None) -> None:
        if self._target is None:
            return
        frame = atoms.copy()
        if energy is not None or forces is not None:
            from ase.calculators.singlepoint import SinglePointCalculator

            kw = {k: v for k, v in (("energy", None if energy is None else float(energy)),
                                    ("forces", None if forces is None else np.asarray(forces, dtype=np.float64)),
                                    ("stress", None if stress is None else np.asarray(stress, dtype=np.float64))) if v is not None}
            frame.calc = SinglePointCalculator(frame, **kw)
        if self._traj is not None:
            self._traj.write(frame)
        elif self._path is not None:
            from ase.io import write as ase_write

            ase_write(str(self._path), frame, append=(self.count > 0))
        else:
            self._target.write(frame)
        self.count += 1

    def close(self) -> None:
        if self._owns and self._traj is not None:
            self._traj.close()
