"""Pure-NumPy two-loop L-BFGS optimizer and vectorized analytical-gradient 3D molecular force field."""

from __future__ import annotations

import math

import numpy as np


def _perp_unit(u: np.ndarray, hint: np.ndarray | None = None) -> np.ndarray:
    if hint is not None:
        p = hint - float(np.dot(hint, u)) * u
        pn = float(np.linalg.norm(p))
        if pn > 1e-5:
            return p / pn
    axis = np.array([1.0, 0.0, 0.0]) if abs(float(u[0])) < 0.8 else np.array([0.0, 1.0, 0.0])
    p = np.cross(u, axis)
    return p / float(np.linalg.norm(p))


def place_terminal_h(
    pos: np.ndarray, placed: list[bool], numbers: np.ndarray, hyb: list[str],
    adj: list[list[int]], r0_map: dict[tuple[int, int], float], angle_fn,
) -> None:
    """Place unplaced terminal hydrogens onto embedded scaffold atoms via VSEPR hybridization geometry."""
    for p in range(len(numbers)):
        if not placed[p]:
            continue
        h_nbrs = [h for h in adj[p] if not placed[h]]
        if not h_nbrs:
            continue
        p_nbrs = [q for q in adj[p] if placed[q]]
        U = [(pos[q] - pos[p]) / max(float(np.linalg.norm(pos[q] - pos[p])), 1e-8) for q in p_nbrs]
        k, m = len(U), len(h_nbrs)
        if k >= 3:
            s = sum(U)
            sn = float(np.linalg.norm(s))
            dirs = [-s / sn if sn > 1e-4 else _perp_unit(U[0], np.cross(U[1] - U[0], U[2] - U[0]))] * m
        elif k == 2:
            s = U[0] + U[1]
            sn = float(np.linalg.norm(s))
            b = -s / sn if sn > 1e-4 else _perp_unit(U[0])
            np_vec = _perp_unit(b, np.cross(U[0], U[1]))
            if m == 1:
                dirs = [b if hyb[p] == "sp2" else math.cos(0.95) * b + math.sin(0.95) * np_vec]
            else:
                dirs = [b / math.sqrt(3.0) + math.sqrt(2.0 / 3.0) * np_vec, b / math.sqrt(3.0) - math.sqrt(2.0 / 3.0) * np_vec]
        else:
            q, u0 = p_nbrs[0], U[0]
            q_oth = [r for r in adj[q] if r != p and placed[r]]
            if hyb[p] == "sp2" and len(q_oth) >= 2:
                perp1 = _perp_unit(u0, np.cross(np.cross(pos[q_oth[0]] - pos[q], pos[q_oth[1]] - pos[q]), u0))
            else:
                perp1 = _perp_unit(u0, -(pos[q_oth[0]] - pos[q]) if q_oth else None)
            perp2 = np.cross(u0, perp1)
            if m == 1:
                th = angle_fn(int(numbers[p]), hyb[p], 0, 0)
                dirs = [math.cos(th) * u0 + math.sin(th) * perp1]
            elif m == 2 and hyb[p] == "sp2":
                th = math.radians(120.0)
                dirs = [math.cos(th) * u0 + math.sin(th) * perp1, math.cos(th) * u0 - math.sin(th) * perp1]
            elif m == 2:
                th = math.radians(107.5)
                dirs = [math.cos(th) * u0 + math.sin(th) * (0.5 * perp1 + sgn * 0.8660254 * perp2) for sgn in (1.0, -1.0)]
            else:
                th = math.acos(-1.0 / 3.0)
                dirs = [math.cos(th) * u0 + math.sin(th) * (math.cos(2.0 * math.pi * j / m) * perp1 + math.sin(2.0 * math.pi * j / m) * perp2) for j in range(m)]
        for h, d_vec in zip(h_nbrs, dirs):
            pos[h] = pos[p] + r0_map[(min(p, h), max(p, h))] * (d_vec / max(float(np.linalg.norm(d_vec)), 1e-8))
            placed[h] = True


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def lbfgs_minimize(x0: np.ndarray, fn, max_iter: int = 95, gtol: float = 1e-2, m_hist: int = 8) -> np.ndarray:
    """Minimize `fn(x) -> (energy, grad)` using pure-NumPy two-loop L-BFGS with Armijo backtracking."""
    x = x0.copy()
    f, g = fn(x)
    s_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rho_list: list[float] = []
    for _ in range(max_iter):
        if float(np.max(np.abs(g))) <= gtol:
            break
        q = g.copy()
        k = len(s_list)
        alphas = [0.0] * k
        for idx in range(k - 1, -1, -1):
            a = rho_list[idx] * float(np.dot(s_list[idx], q))
            alphas[idx] = a
            q -= a * y_list[idx]
        if k > 0:
            gamma = float(np.dot(s_list[-1], y_list[-1])) / max(float(np.dot(y_list[-1], y_list[-1])), 1e-16)
            r = gamma * q
        else:
            r = 0.01 * q
        for idx in range(k):
            r += s_list[idx] * (alphas[idx] - rho_list[idx] * float(np.dot(y_list[idx], r)))
        d = -r
        dg = float(np.dot(d, g))
        if dg >= 0.0:
            s_list.clear()
            y_list.clear()
            rho_list.clear()
            d = -0.01 * g
            dg = float(np.dot(d, g))
        max_step = float(np.max(np.abs(d)))
        if max_step > 0.40:
            d *= 0.40 / max_step
            dg = float(np.dot(d, g))
        step = 1.0
        x_new, f_new, g_new = x, f, g
        accepted = False
        for _ in range(8):
            x_try = x + step * d
            f_try, g_try = fn(x_try)
            if f_try <= f + 1e-4 * step * dg:
                x_new, f_new, g_new, accepted = x_try, f_try, g_try, True
                break
            step *= 0.4
        if not accepted:
            x_new = x + step * d
            f_new, g_new = fn(x_new)
        s, y = x_new - x, g_new - g
        sy = float(np.dot(s, y))
        if sy > 1e-12:
            if len(s_list) >= m_hist:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)
        x, f, g = x_new, f_new, g_new
    return x


def make_forcefield(
    n: int,
    b_idx: np.ndarray,
    b_r0: np.ndarray,
    a_idx: np.ndarray,
    a_th0: np.ndarray,
    a_lin: np.ndarray,
    imp_idx: np.ndarray,
    ch_idx: np.ndarray,
    ch_sgn: np.ndarray,
    ez_idx: np.ndarray,
    ez_sgn: np.ndarray,
    sp2_idx: np.ndarray,
    sp3_idx: np.ndarray,
    nb_idx: np.ndarray,
    nb_r0: np.ndarray,
    nb_rc: np.ndarray,
):
    """Build a fast vectorized `energy_and_grad(flat_coords) -> (energy, flat_grad)` closure."""
    has_b, has_a, has_imp, has_ch, has_nb = len(b_idx) > 0, len(a_idx) > 0, len(imp_idx) > 0, len(ch_idx) > 0, len(nb_idx) > 0
    any_lin, any_nl = bool(np.any(a_lin)), bool(np.any(~a_lin))
    nl_mask = ~a_lin
    tors_active = [(ti, m) for ti, m in ((ez_idx, 0), (sp2_idx, 1), (sp3_idx, 2)) if len(ti) > 0]

    def energy_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
        pos = flat.reshape(n, 3)
        idx_parts: list[np.ndarray] = []
        vec_parts: list[np.ndarray] = []
        e_tot = 0.0

        if has_b:
            d = pos[b_idx[:, 0]] - pos[b_idx[:, 1]]
            r = np.sqrt((d * d).sum(axis=1) + 1e-16)
            dr = r - b_r0
            e_tot += float(150.0 * np.dot(dr, dr))
            g = (300.0 * (dr / r))[:, None] * d
            idx_parts.extend((b_idx[:, 0], b_idx[:, 1]))
            vec_parts.extend((g, -g))

        if has_a:
            u_vec = pos[a_idx[:, 0]] - pos[a_idx[:, 1]]
            v_vec = pos[a_idx[:, 2]] - pos[a_idx[:, 1]]
            ru = np.sqrt((u_vec * u_vec).sum(axis=1) + 1e-16)
            rv = np.sqrt((v_vec * v_vec).sum(axis=1) + 1e-16)
            uh, vh = u_vec / ru[:, None], v_vec / rv[:, None]
            c = np.clip((uh * vh).sum(axis=1), -0.999999, 0.999999)
            de_dc = np.empty_like(c)
            if any_lin:
                e_tot += float(40.0 * np.sum(1.0 + c[a_lin]))
                de_dc[a_lin] = 40.0
            if any_nl:
                cn = c[nl_mask]
                dth = np.arccos(cn) - a_th0[nl_mask]
                e_tot += float(35.0 * np.dot(dth, dth))
                de_dc[nl_mask] = -70.0 * dth / np.sqrt(np.maximum(1.0 - cn * cn, 1e-12))
            gi = de_dc[:, None] * (vh - c[:, None] * uh) / ru[:, None]
            gk = de_dc[:, None] * (uh - c[:, None] * vh) / rv[:, None]
            idx_parts.extend((a_idx[:, 0], a_idx[:, 2], a_idx[:, 1]))
            vec_parts.extend((gi, gk, -(gi + gk)))

        if has_imp:
            u = pos[imp_idx[:, 1]] - pos[imp_idx[:, 0]]
            v = pos[imp_idx[:, 2]] - pos[imp_idx[:, 0]]
            w = pos[imp_idx[:, 3]] - pos[imp_idx[:, 0]]
            vw, wu, uv = _cross3(v, w), _cross3(w, u), _cross3(u, v)
            vol = (u * vw).sum(axis=1)
            e_tot += float(30.0 * np.dot(vol, vol))
            s = (60.0 * vol)[:, None]
            gu, gv, gw = s * vw, s * wu, s * uv
            idx_parts.extend((imp_idx[:, 1], imp_idx[:, 2], imp_idx[:, 3], imp_idx[:, 0]))
            vec_parts.extend((gu, gv, gw, -(gu + gv + gw)))

        if has_ch:
            u = pos[ch_idx[:, 0]] - pos[ch_idx[:, 3]]
            v = pos[ch_idx[:, 1]] - pos[ch_idx[:, 3]]
            w = pos[ch_idx[:, 2]] - pos[ch_idx[:, 3]]
            vw = _cross3(v, w)
            viol = 2.0 - ch_sgn * (u * vw).sum(axis=1)
            act = viol > 0.0
            if np.any(act):
                wu, uv = _cross3(w[act], u[act]), _cross3(u[act], v[act])
                e_tot += float(25.0 * np.dot(viol[act], viol[act]))
                s = (-50.0 * viol[act] * ch_sgn[act])[:, None]
                gu, gv, gw = s * vw[act], s * wu, s * uv
                ia = ch_idx[act]
                idx_parts.extend((ia[:, 0], ia[:, 1], ia[:, 2], ia[:, 3]))
                vec_parts.extend((gu, gv, gw, -(gu + gv + gw)))

        for t_idx, mode in tors_active:
            b1 = pos[t_idx[:, 1]] - pos[t_idx[:, 0]]
            b2 = pos[t_idx[:, 2]] - pos[t_idx[:, 1]]
            b3 = pos[t_idx[:, 3]] - pos[t_idx[:, 2]]
            m, nn = _cross3(b1, b2), _cross3(b2, b3)
            rm = np.sqrt((m * m).sum(axis=1) + 1e-14)
            rn = np.sqrt((nn * nn).sum(axis=1) + 1e-14)
            mh, nh = m / rm[:, None], nn / rn[:, None]
            c = np.clip((mh * nh).sum(axis=1), -1.0, 1.0)
            if mode == 0:
                e_tot += float(18.0 * np.sum(1.0 - ez_sgn * c))
                de_dc = -18.0 * ez_sgn
            elif mode == 1:
                e_tot += float(8.0 * np.sum(1.0 - c * c))
                de_dc = -16.0 * c
            else:
                e_tot += float(0.4 * np.sum(1.0 + 4.0 * c * c * c - 3.0 * c))
                de_dc = 0.4 * (12.0 * c * c - 3.0)
            gm = de_dc[:, None] * (nh - c[:, None] * mh) / rm[:, None]
            gn = de_dc[:, None] * (mh - c[:, None] * nh) / rn[:, None]
            gb1, gb3 = _cross3(b2, gm), _cross3(gn, b2)
            gb2 = _cross3(gm, b1) + _cross3(b3, gn)
            idx_parts.extend((t_idx[:, 0], t_idx[:, 1], t_idx[:, 2], t_idx[:, 3]))
            vec_parts.extend((-gb1, gb1 - gb2, gb2 - gb3, gb3))

        if has_nb:
            d = pos[nb_idx[:, 0]] - pos[nb_idx[:, 1]]
            r = np.sqrt((d * d).sum(axis=1) + 1e-12)
            pen = nb_r0 - r
            act = pen > 0.0
            if np.any(act):
                pa, ra, da = pen[act], r[act], d[act]
                e_tot += float(12.0 * np.dot(pa, pa))
                f_mag = -24.0 * pa
                core = nb_rc[act] - ra
                cm = core > 0.0
                if np.any(cm):
                    c2 = core[cm] * core[cm]
                    e_tot += float(80.0 * np.dot(c2, c2))
                    f_mag[cm] -= 320.0 * c2 * core[cm]
                g = (f_mag / ra)[:, None] * da
                ia_nb = nb_idx[act]
                idx_parts.extend((ia_nb[:, 0], ia_nb[:, 1]))
                vec_parts.extend((g, -g))

        if not idx_parts:
            return e_tot, np.zeros(3 * n, dtype=np.float64)
        all_idx = np.concatenate(idx_parts)
        all_vec = np.concatenate(vec_parts, axis=0)
        gx = np.bincount(all_idx, weights=all_vec[:, 0], minlength=n)
        gy = np.bincount(all_idx, weights=all_vec[:, 1], minlength=n)
        gz = np.bincount(all_idx, weights=all_vec[:, 2], minlength=n)
        return e_tot, np.column_stack((gx, gy, gz)).reshape(-1)

    return energy_and_grad
