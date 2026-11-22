"""FP64 coordinate recentering, charged-dipole restoration, and safe ASE L-BFGS guards."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
import warnings

from ase import Atoms
from ase.constraints import FixAtoms
import numpy as np

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator


def center_positions_f64(pos_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the structure barycenter in `float64` before casting coordinates to `float32`."""
    p64 = np.asarray(pos_np, dtype=np.float64)
    if len(p64) == 0:
        return np.ascontiguousarray(p64, dtype=np.float64), np.zeros(3, dtype=np.float64)
    origin = np.ascontiguousarray(p64.mean(axis=0), dtype=np.float64)
    return np.ascontiguousarray(p64 - origin, dtype=np.float64), origin


def restore_charged_dipole_f64(
    dip_ea: np.ndarray,
    q_total: float,
    origin_f64: np.ndarray,
    has_dipole: bool,
) -> np.ndarray:
    """Restore `+ Q_tot * origin_f64` (e*A) on open charged systems in `float64`."""
    if has_dipole and q_total != 0.0:
        return np.asarray(dip_ea, dtype=np.float64) + float(q_total) * np.asarray(origin_f64, dtype=np.float64)
    return np.array(dip_ea, dtype=np.float64, copy=True)


def run_recentered_relax_batch(
    calc: "MenaCalculator",
    atoms_list: Sequence[Atoms],
    *,
    fmax: float = 0.01,
    steps: int = 200,
    maxstep: float = 0.2,
    skin: float = 0.6,
    history_size: int = 8,
    max_atoms: int = 3072,
    max_batch: int = 64,
) -> list[dict[str, Any]]:
    """Center each structure in `float64` around its barycenter before `relax_batch_frames`, then restore coordinates."""
    from lam4.calc.relax import relax_batch_frames

    n_tot = len(atoms_list)
    if n_tot == 0:
        return []

    centered_list: list[Atoms] = []
    origins: list[np.ndarray] = []
    fixed_indices_list: list[np.ndarray | None] = []
    orig_positions_list: list[np.ndarray] = []

    for at in atoms_list:
        pos64 = np.asarray(at.get_positions(), dtype=np.float64).copy()
        pos_c, origin = center_positions_f64(pos64)
        at_c = at.copy()
        at_c.calc = None
        saved_cons = list(at_c.constraints)
        at_c.constraints = []
        at_c.set_positions(pos_c)
        at_c.constraints = saved_cons
        if "restraint_ref" in at_c.arrays:
            at_c.arrays["restraint_ref"] = np.asarray(at_c.arrays["restraint_ref"], dtype=np.float64) - origin
        if "restraint_ref" in at_c.info and at_c.info["restraint_ref"] is not None:
            at_c.info["restraint_ref"] = np.asarray(at_c.info["restraint_ref"], dtype=np.float64) - origin
        fixed_idx: np.ndarray | None = None
        cons = list(getattr(at, "constraints", ()) or ())
        if cons and all(isinstance(c, FixAtoms) for c in cons):
            idx_parts = [np.asarray(c.get_indices(), dtype=np.int64) for c in cons]
            cat = np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=np.int64)
            if len(cat) > 0:
                fixed_idx = cat
        centered_list.append(at_c)
        origins.append(origin)
        fixed_indices_list.append(fixed_idx)
        orig_positions_list.append(pos64)

    outs = relax_batch_frames(
        calc, centered_list, fmax=fmax, steps=steps, maxstep=maxstep,
        skin=skin, history_size=history_size, max_atoms=max_atoms, max_batch=max_batch,
    )
    for orig_at, orig_pos, origin, fixed_idx, res in zip(atoms_list, orig_positions_list, origins, fixed_indices_list, outs):
        p_restored = np.asarray(res["positions"], dtype=np.float64) + origin
        if fixed_idx is not None:
            p_restored[fixed_idx] = orig_pos[fixed_idx]
        at_out = orig_at.copy()
        at_out.calc = None
        saved_out_cons = list(at_out.constraints)
        at_out.constraints = []
        at_out.set_positions(p_restored)
        at_out.constraints = saved_out_cons
        res["atoms"] = at_out
        res["positions"] = p_restored
    return outs


def run_safe_ase_relax(
    opt: Any,
    target: Any,
    *,
    fmax: float,
    steps: int,
    calc: Any,
) -> tuple[bool, float]:
    """Run an ASE optimizer while guarding `LBFGS`/`BFGS` against `y*s == 0` division-by-zero and NaN steps."""
    opt_state = {"hits": 0, "stalled": False}
    wrapped_states: set[int] = set()

    def _wrap_state_update() -> None:
        st = getattr(opt, "state", None)
        if st is None or id(st) in wrapped_states or not hasattr(st, "update"):
            return
        wrapped_states.add(id(st))
        orig_update = st.update

        def _safe_update(pos: np.ndarray, gradient: np.ndarray, pos0: np.ndarray, gradient0: np.ndarray) -> None:
            s0 = pos - pos0
            y0 = gradient - gradient0
            ys = float(np.dot(y0, s0))
            if not np.isfinite(ys) or abs(ys) <= 1e-24:
                opt_state["hits"] += 1
                if hasattr(st, "s") and hasattr(st, "y") and hasattr(st, "rho"):
                    st.s.clear()
                    st.y.clear()
                    st.rho.clear()
                    st.iteration = 0
                elif hasattr(st, "hessian") and hasattr(opt, "H0"):
                    st.hessian = opt.H0.copy()
                return
            opt_state["hits"] = 0
            orig_update(pos, gradient, pos0, gradient0)

        st.update = _safe_update

    _wrap_state_update()
    optim = opt.optimizable
    best_x = np.asarray(optim.get_x(), dtype=np.float64).copy()
    best_fmax = float("inf")
    orig_step = opt.step

    def _safe_step(*args: Any, **kwargs: Any) -> Any:
        _wrap_state_update()
        x_before = np.asarray(optim.get_x(), dtype=np.float64).copy()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning, message=r".*(divide by zero|invalid value).*")
                out = orig_step(*args, **kwargs)
        except (FloatingPointError, ZeroDivisionError, RuntimeWarning):
            opt_state["stalled"] = True
            optim.set_x(best_x)
            return None
        _wrap_state_update()
        x_after = np.asarray(optim.get_x(), dtype=np.float64)
        if (not np.all(np.isfinite(x_after))) or np.allclose(x_after, x_before, atol=1e-15, rtol=0.0) or opt_state["hits"] >= 3:
            opt_state["stalled"] = True
            optim.set_x(best_x)
            return out
        return out

    opt.step = _safe_step
    converged = False
    for is_conv in opt.irun(fmax=float(fmax), steps=int(steps)):
        f_cur = np.asarray(target.get_forces(), dtype=np.float64)
        fm_cur = float(np.sqrt((f_cur ** 2).sum(axis=-1)).max()) if len(f_cur) else 0.0
        x_cur = np.asarray(optim.get_x(), dtype=np.float64)
        if np.isfinite(fm_cur) and np.all(np.isfinite(x_cur)) and fm_cur <= best_fmax:
            best_fmax = fm_cur
            best_x = x_cur.copy()
        converged = bool(is_conv)
        if converged:
            break
        if opt_state["stalled"]:
            optim.set_x(best_x)
            converged = bool(best_fmax <= float(fmax))
            if not converged and getattr(calc, "precision", "auto") != "exact":
                warnings.warn(
                    f"{type(opt).__name__} stagnated near {getattr(calc, 'precision', 'auto')} precision floor "
                    f"(fmax={best_fmax:.2e} > {float(fmax):.2e} eV/A); pass precision='exact' for tighter relaxation.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            break

    return converged, (best_fmax if np.isfinite(best_fmax) else 0.0)
