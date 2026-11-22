// Geometry helpers for the viewer: bonds by a cell-list search, periodic repeats, and cached shaded atom sprites.
import { color, radius } from './elements.js';

export const MAX_ATOMS = 5000;      // most atoms drawn (repeats are reduced to stay below this)
export const MAX_BONDS_ATOMS = 4000; // above this, bonds are skipped
export const MAX_LABEL_ATOMS = 150;

// Bonds inside the given set of positions (periodic images are not searched: bonds may skip the cell edge).
export function bondsOf(symbols, pos) {
  const n = symbols.length, out = [];
  if (n > MAX_BONDS_ATOMS) return out;
  const h = 3.4, grid = new Map(), key = (i, j, k) => i + ',' + j + ',' + k;
  const rad = symbols.map(radius);
  for (let a = 0; a < n; a++) {
    const p = pos[a], k = key(Math.floor(p[0] / h), Math.floor(p[1] / h), Math.floor(p[2] / h));
    let b = grid.get(k); if (!b) grid.set(k, b = []); b.push(a);
  }
  for (let a = 0; a < n; a++) {
    const p = pos[a], ci = Math.floor(p[0] / h), cj = Math.floor(p[1] / h), ck = Math.floor(p[2] / h);
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) for (let dk = -1; dk <= 1; dk++) {
      const bucket = grid.get(key(ci + di, cj + dj, ck + dk)); if (!bucket) continue;
      for (const b of bucket) {
        if (b <= a) continue;
        const q = pos[b], d = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
        if (d > 0.4 && d < 1.2 * (rad[a] + rad[b])) out.push([a, b]);
      }
    }
  }
  return out;
}

// Largest repeat r (≤ want) with n·r³ atoms under the drawing limit.
export function safeRepeat(n, want) {
  let r = Math.max(1, want | 0);
  while (r > 1 && n * r ** 3 > MAX_ATOMS) r--;
  return r;
}

// Copies of per-atom rows for an r×r×r block of cells: positions are translated, vectors (forces, modes) are copied.
export function expand(rows, cell, r, translate = true) {
  if (!cell || r <= 1 || !rows) return rows;
  const out = [];
  for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) for (let k = 0; k < r; k++) {
    const t = [0, 1, 2].map(c => i * cell[0][c] + j * cell[1][c] + k * cell[2][c]);
    for (const p of rows) out.push(translate ? [p[0] + t[0], p[1] + t[1], p[2] + t[2]] : p);
  }
  return out;
}

export function scaleCell(cell, r) { return cell ? cell.map(v => v.map(x => x * r)) : null; }

// Pre-rendered atom discs (cached by colour/element, pixel radius, and DPR): drawImage is far cheaper than a gradient per atom.
const sprites = new Map();

function quantizeHex(hex) {
  if (!hex || hex[0] !== '#' || hex.length < 7) return '#94a3b8';
  const n = parseInt(hex.slice(1, 7), 16);
  const q = s => Math.min(255, Math.round(((n >> s) & 255) / 4) * 4).toString(16).padStart(2, '0');
  return '#' + q(16) + q(8) + q(0);
}

function makeSprite(keyS, col, r, flat, dpr) {
  let c = sprites.get(keyS);
  if (c) return c;
  if (sprites.size > 1200) sprites.clear();
  const size = Math.ceil((2 * r + 2) * dpr);
  c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d'), m = size / 2, R = r * dpr;
  if (flat) { g.fillStyle = col; }
  else {
    const grd = g.createRadialGradient(m - R * .35, m - R * .35, R * .08, m, m, R);
    grd.addColorStop(0, mix(col, '#ffffff', .55)); grd.addColorStop(.35, col); grd.addColorStop(1, mix(col, '#000000', .38));
    g.fillStyle = grd;
  }
  g.beginPath(); g.arc(m, m, R, 0, Math.PI * 2); g.fill();
  g.lineWidth = Math.max(1, dpr * .8); g.strokeStyle = 'rgba(0,0,0,.28)'; g.stroke();
  c.r = r; sprites.set(keyS, c);
  return c;
}

export function sprite(sym, px, flat, dprOverride) {
  const r = Math.max(2, Math.round(px)), dpr = dprOverride || window.devicePixelRatio || 1;
  return makeSprite(sym + ':' + r + ':' + dpr + (flat ? 'f' : ''), color(sym), r, flat, dpr);
}

export function spriteColor(hex, px, flat, dprOverride) {
  const r = Math.max(2, Math.round(px)), dpr = dprOverride || window.devicePixelRatio || 1, qh = quantizeHex(hex);
  return makeSprite(qh + ':' + r + ':' + dpr + (flat ? 'f' : ''), qh, r, flat, dpr);
}

export function mix(hex, other, t) {
  const a = parseInt(hex.slice(1), 16), b = parseInt(other.slice(1), 16);
  const ch = (x, y) => Math.round(x + (y - x) * t);
  return `rgb(${ch(a >> 16, b >> 16)},${ch((a >> 8) & 255, (b >> 8) & 255)},${ch(a & 255, b & 255)})`;
}

export function mixHex(a, b, t) {
  const u = Math.max(0, Math.min(1, t));
  const x = parseInt(a.slice(1), 16), y = parseInt(b.slice(1), 16);
  const ch = s => Math.round(((x >> s) & 255) + (((y >> s) & 255) - ((x >> s) & 255)) * u).toString(16).padStart(2, '0');
  return '#' + ch(16) + ch(8) + ch(0);
}

// Scientific colormaps: electrostatic convention red = negative (δ−, electron-rich), blue = positive (δ+, electron-poor).
export const COLORMAPS = {
  charges: { label: 'Charge q (e)', unit: 'e', stops: ['#b91c1c', '#f1f5f9', '#1d4ed8'], diverging: true },
  charge:  { label: 'Charge q (e)', unit: 'e', stops: ['#b91c1c', '#f1f5f9', '#1d4ed8'], diverging: true },
  spins:   { label: 'Spin s (e)', unit: 'e', stops: ['#7c3aed', '#f1f5f9', '#0d9488'], diverging: true },
  spin:    { label: 'Spin s (e)', unit: 'e', stops: ['#7c3aed', '#f1f5f9', '#0d9488'], diverging: true },
  unpaired: { label: 'Unpaired electrons (index)', unit: '', stops: ['#f1f5f9', '#0d9488', '#134e4a'], diverging: false, zeroFloor: true, minHi: 0.3 },
  pauli:   { label: 'Pauli repulsion (eV)', unit: 'eV', stops: ['#fef9c3', '#f59e0b', '#b91c1c'], diverging: false, zeroFloor: true },
  alpha:   { label: 'Polarizability α (Å³)', unit: 'Å³', stops: ['#e0e7ff', '#4f46e5', '#1e1b4b'], diverging: false },
  fmag:    { label: 'Force |F| (eV/Å)', unit: 'eV/Å', stops: ['#dcfce7', '#f59e0b', '#dc2626'], diverging: false, zeroFloor: true },
};

export function scalarRange(mode, values) {
  const cm = COLORMAPS[mode] || COLORMAPS.alpha;
  const vs = (values || []).filter(Number.isFinite);
  if (cm.diverging) {
    let m = 0.01;
    for (let i = 0; i < vs.length; i++) { const a = Math.abs(vs[i]); if (a > m) m = a; }
    return { lo: -m, mid: 0, hi: m, diverging: true };
  }
  if (cm.zeroFloor) {
    let m = cm.minHi || 0.01;  // minHi: a closed-shell molecule's noise-level values stay pale instead of filling the scale
    for (let i = 0; i < vs.length; i++) if (vs[i] > m) m = vs[i];
    return { lo: 0, mid: m / 2, hi: m, diverging: false };
  }
  let lo = vs.length ? Infinity : 0, hi = vs.length ? -Infinity : 1;
  for (let i = 0; i < vs.length; i++) { if (vs[i] < lo) lo = vs[i]; if (vs[i] > hi) hi = vs[i]; }
  if (hi - lo < 1e-3) { const c = (lo + hi) / 2; lo = Math.max(0, c - 0.1); hi = c + 0.1; }
  return { lo, mid: (lo + hi) / 2, hi, diverging: false };
}

export function scalarColor(mode, val, lo, hi) {
  const cm = COLORMAPS[mode];
  if (!cm || !Number.isFinite(val)) return '#94a3b8';
  const raw = hi > lo ? (val - lo) / (hi - lo) : 0.5;
  const t = Math.round(Math.max(0, Math.min(1, raw)) * 128) / 128;
  return t <= 0.5 ? mixHex(cm.stops[0], cm.stops[1], t * 2) : mixHex(cm.stops[1], cm.stops[2], (t - 0.5) * 2);
}
