// A 3D structure viewer on a canvas (no libraries): drag to turn, wheel to zoom, frames, forces, dipole, physics colormaps,
// hover atom inspection, periodic repeats, vibration-mode animation, 2× white PNG export, and live position updates.
import { el, clear, fmt, cssVar } from './util.js';
import { color, radius } from './elements.js';
import { bondsOf, expand, safeRepeat, scaleCell, sprite, spriteColor, COLORMAPS, scalarRange, scalarColor, MAX_LABEL_ATOMS } from './viewer_geom.js';

const SCALAR_MODES = ['charges', 'spins', 'unpaired', 'pauli', 'alpha', 'fmag'];

function fmtSigned(v, d = 3) {
  const n = Math.abs(v) < 0.5 * 10 ** -d ? 0 : v;
  return (n < 0 ? '−' : n > 0 ? '+' : '') + Math.abs(n).toFixed(d);
}

function fmtBarVal(v, signed) {
  if (Math.abs(v) < 1e-9) return '0';
  const a = Math.abs(v), s = a >= 10 ? a.toFixed(1) : a >= 0.1 ? a.toFixed(2) : a.toFixed(3);
  return (v < 0 ? '−' : (signed ? '+' : '')) + s;
}

function rrect(g, x, y, w, h, r) {
  g.beginPath();
  if (g.roundRect) g.roundRect(x, y, w, h, r);
  else g.rect(x, y, w, h);
}

export class Viewer {
  constructor(container) {
    this.yaw = 0.55; this.pitch = -0.35; this.zoom = 1; this.rep = 1;
    this.frames = []; this.cell = null; this.modes = null; this.f = 0; this.mode = -1; this.phase = 0;
    this.playing = false; this.interval = 700; this.showForces = false; this.showLabels = false; this.showDipole = true;
    this.colorMode = 'element'; this.colorModes = [{ id: 'element', label: 'Element' }]; this.colorRanges = {};
    this.hoverAtom = -1; this._hit = null;
    this.root = el('div', { class: 'viewer' });
    this.canvas = el('canvas', { tabindex: 0, role: 'img', 'aria-label': '3D view of the structure. Drag or use the arrow keys to turn, scroll or + and - to zoom.' });
    this.label = el('div', { class: 'vlabel' });
    this.inspect = el('div', { class: 'vinspect', hidden: true });
    this.legend = el('div', { class: 'vlegend', 'aria-hidden': 'true' });
    this.empty = el('div', { class: 'vempty' }, 'Choose an example or drop a file to see the structure here.');
    this.bar = el('div', { class: 'vbar' });
    this.root.append(el('div', { class: 'vstage' }, this.canvas, this.label, this.inspect, this.legend, this.empty,
      el('div', { class: 'vhelp' }, 'drag to turn · scroll to zoom')), this.bar);
    container.append(this.root);
    this.bind();
    this.ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => this.schedule()) : null;
    if (this.ro) this.ro.observe(this.canvas);
    this.schedule();
  }

  rawScalars(fr, mode) {
    if (!fr || !fr.symbols) return null;
    const n = fr.symbols.length;
    if (mode === 'fmag') {
      if (fr.fmag && fr.fmag.length === n) return fr.fmag;
      if (fr.forces && fr.forces.length === n) return fr._fmag || (fr._fmag = fr.forces.map(v => Math.hypot(v[0], v[1], v[2])));
      return null;
    }
    const arr = fr[mode];
    return arr && arr.length === n ? arr : null;
  }

  frameScalars(fr, mode) {
    const raw = this.rawScalars(fr, mode);
    if (raw) return raw;
    if (fr && fr.symbols && (mode === 'charges' || mode === 'spins') && this.colorRanges[mode]) return new Array(fr.symbols.length).fill(0);
    return null;
  }

  computeColorMeta() {
    this.colorRanges = {};
    const modes = [{ id: 'element', label: 'Element' }];
    for (const m of SCALAR_MODES) {
      const vals = [];
      let any = false;
      for (const fr of this.frames) {
        const arr = this.rawScalars(fr, m);
        if (arr) { any = true; for (const v of arr) if (Number.isFinite(v)) vals.push(v); }
      }
      if (any) {
        this.colorRanges[m] = scalarRange(m, vals);
        modes.push({ id: m, label: COLORMAPS[m].label });
      }
    }
    this.colorModes = modes;
    if (!modes.some(m => m.id === this.colorMode)) this.colorMode = 'element';
  }

  // data = {frames: [{symbols, positions, forces?, fmag?, charges?, spins?, alpha?, dipole?, label?}], cell, modes?}
  show(data, opts = {}) {
    this.stop();
    this.frames = (data && data.frames) || []; this.cell = (data && data.cell) || null; this.modes = (data && data.modes && data.modes.length) ? data.modes : null;
    this.f = opts.frame === 'last' ? Math.max(0, this.frames.length - 1) : 0; this.mode = -1; this.bonds = new Map(); this._fit = null;
    this.hoverAtom = -1; this._hit = null; this.inspect.hidden = true;
    const n = this.frames.length ? this.frames[0].symbols.length : 0;
    if (!opts.keepRepeat) this.rep = this.cell ? safeRepeat(n, n <= 4 ? 3 : n <= 40 ? 2 : 1) : 1;
    this.rep = this.cell ? safeRepeat(n, this.rep) : 1;
    this.empty.hidden = this.frames.length > 0;
    this.computeColorMeta();
    this.buildBar(); this.buildLegend(); this.schedule();
  }
  // Positions only (a running job): replaces the current frame and keeps the camera. If the worker repeated the cell, updates symbols & cell too.
  live(positions, label, frame) {
    const syms = (frame && frame.symbols) || (this.frames.length ? this.frames[this.frames.length - 1].symbols : null);
    if (!syms || !positions || positions.length !== syms.length) return;
    if (frame && frame.cell !== undefined) this.cell = frame.cell;
    const f0 = this.frames[0];
    if (frame && frame.symbols && (!this.frames.length || f0.symbols.length !== syms.length)) {
      this.rep = 1;
      this._fit = null;
      this.frames = [{ symbols: syms, positions, label, cell: this.cell }];
      this.f = 0;
      this.empty.hidden = true;
      this.computeColorMeta(); this.buildBar(); this.buildLegend();
    } else if (this.frames.length > 1 || (f0 && (f0.forces || f0.fmag || f0.charges || f0.spins || f0.alpha))) {
      this.frames = [{ symbols: syms, positions, label, cell: this.cell }];
      this.f = 0;
      this.computeColorMeta(); this.buildBar(); this.buildLegend();
    } else {
      f0.positions = positions; f0.label = label; f0.cell = this.cell;
    }
    this.bonds = new Map(); this.schedule();
  }
  clear() { this.show({ frames: [], cell: null }); }
  // Colour the atoms by one of the available modes (from the interpretation panel); ignored if the frame lacks it.
  setColorMode(id) {
    if (!this.colorModes.some(m => m.id === id)) return;
    this.colorMode = id;
    if (this.colorSel) this.colorSel.value = id;
    this.legend.hidden = id !== 'element';
    this.schedule();
  }

  buildBar() {
    clear(this.bar);
    const fr = this.frames[0];
    if (!fr) { this.bar.hidden = true; return; }
    this.bar.hidden = false;
    if (this.frames.length > 1) {
      this.playBtn = el('button', { class: 'btn small', type: 'button', 'aria-label': 'Play', onclick: () => this.toggle() }, 'Play');
      this.slider = el('input', { type: 'range', min: 0, max: this.frames.length - 1, value: this.f, 'aria-label': 'Frame', oninput: () => { this.f = +this.slider.value; this.updateInspect(); this.schedule(); } });
      this.count = el('span', { class: 'vcount' });
      this.bar.append(el('button', { class: 'btn small', type: 'button', 'aria-label': 'Previous frame', onclick: () => this.step(-1) }, '‹'), this.playBtn,
        el('button', { class: 'btn small', type: 'button', 'aria-label': 'Next frame', onclick: () => this.step(1) }, '›'), this.slider, this.count);
    } else this.slider = null;
    if (this.colorModes && this.colorModes.length > 1) {
      const sel = el('select', { 'aria-label': 'Color atoms by', onchange: () => {
        this.colorMode = sel.value;
        this.legend.hidden = this.colorMode !== 'element';
        this.schedule();
      } }, ...this.colorModes.map(m => el('option', { value: m.id, selected: m.id === this.colorMode }, m.label)));
      this.colorSel = sel;
      this.bar.append(el('label', { class: 'vctl' }, 'Color ', sel));
    } else this.colorSel = null;
    if (this.cell) {
      const n = fr.symbols.length;
      const sel = el('select', { 'aria-label': 'Repeat the cell', onchange: () => { this.rep = +sel.value; this.bonds = new Map(); this._fit = null; this.hoverAtom = -1; this.updateInspect(); this.schedule(); } },
        ...[1, 2, 3, 4].filter(r => r === 1 || n * r ** 3 <= 5000).map(r => el('option', { value: r, selected: r === this.rep }, r === 1 ? 'one cell' : `${r}×${r}×${r} cells`)));
      this.bar.append(el('label', { class: 'vctl' }, 'Show ', sel));
    }
    if (this.frames.some(q => q.forces)) this.bar.append(this.check('Forces', this.showForces, v => { this.showForces = v; this.schedule(); }));
    if (this.frames.some(q => q.dipole && Math.hypot(...q.dipole) >= 0.02)) this.bar.append(this.check('Dipole', this.showDipole, v => { this.showDipole = v; this.schedule(); }));
    if (fr.symbols.length <= MAX_LABEL_ATOMS) this.bar.append(this.check('Indices', this.showLabels, v => { this.showLabels = v; this.schedule(); }));
    if (this.modes) {
      const sel = el('select', { 'aria-label': 'Vibration to animate', onchange: () => { this.mode = +sel.value; this.phase = 0; this.mode >= 0 ? this.animate() : this.schedule(); } },
        el('option', { value: -1 }, 'No vibration'),
        ...this.modes.map((m, i) => el('option', { value: i }, `${i + 1}: ${m.freq < 0 ? fmt(-m.freq, 1) + 'i' : fmt(m.freq, 1)} cm⁻¹${m.freq < 0 ? ' (imaginary)' : (m.ir !== undefined ? ` · IR ${fmt(m.ir, 1)} km/mol` : '')}`)));
      this.modeSel = sel;
      this.bar.append(el('label', { class: 'vctl' }, 'Animate ', sel));
    }
    this.bar.append(
      el('button', { class: 'btn ghost small vreset', type: 'button', title: 'Reset camera angle and zoom', onclick: () => { this.yaw = 0.55; this.pitch = -0.35; this.zoom = 1; this.schedule(); } }, 'Reset'),
      el('button', { class: 'btn ghost small', type: 'button', title: 'Save a 2× PNG image on a white background', onclick: () => this.exportPng() }, 'PNG'),
    );
  }
  setMode(i) { if (!this.modeSel) return; this.modeSel.value = i; this.modeSel.dispatchEvent(new Event('change')); }
  buildLegend() {
    clear(this.legend);
    this.legend.hidden = this.colorMode !== 'element';
    const syms = this.frames[0] ? [...new Set(this.frames[0].symbols)] : [];
    if (syms.length > 12) return;
    syms.forEach(s => this.legend.append(el('span', {}, el('i', { style: `background:${color(s)}` }), s)));
  }
  check(text, on, fn) {
    const c = el('input', { type: 'checkbox', checked: on, onchange: () => fn(c.checked) });
    return el('label', { class: 'vctl check' }, c, text);
  }

  updateInspect() {
    const fr = this.frames[this.f];
    if (this.hoverAtom < 0 || !fr || !this._hit || this.hoverAtom >= this._hit.sym.length) {
      this.inspect.hidden = true;
      return;
    }
    const a = this.hoverAtom, nb = fr.symbols.length, bi = a % nb, s = this._hit.sym[a];
    const parts = [`#${bi + 1} ${s}`];
    if (fr.charges && fr.charges.length === nb) parts.push(`q = ${fmtSigned(fr.charges[bi], 3)} e`);
    if (fr.spins && fr.spins.length === nb) parts.push(`s = ${fmtSigned(fr.spins[bi], 3)} e`);
    if (fr.alpha && fr.alpha.length === nb) parts.push(`α = ${Number(fr.alpha[bi]).toFixed(2)} Å³`);
    if (fr.unpaired && fr.unpaired.length === nb) parts.push(`unpaired = ${Number(fr.unpaired[bi]).toFixed(3)}`);
    if (fr.pauli && fr.pauli.length === nb) parts.push(`Pauli = ${Number(fr.pauli[bi]).toFixed(3)} eV`);
    const fm = this.rawScalars(fr, 'fmag');
    if (fm && fm.length === nb) parts.push(`|F| = ${Number(fm[bi]).toFixed(3)} eV/Å`);
    if (parts.length === 1 && this._hit.pos && this._hit.pos[a]) {
      const p = this._hit.pos[a];
      parts.push(`(${p[0].toFixed(2)}, ${p[1].toFixed(2)}, ${p[2].toFixed(2)}) Å`);
    }
    this.inspect.textContent = parts.join(' · ');
    this.inspect.hidden = false;
  }

  bind() {
    const c = this.canvas; let drag = null;
    c.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY }; c.setPointerCapture(e.pointerId); });
    c.addEventListener('pointermove', e => {
      if (drag) {
        this.yaw += (e.clientX - drag.x) * .01; this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + (e.clientY - drag.y) * .01));
        drag = { x: e.clientX, y: e.clientY }; this.schedule();
        return;
      }
      if (!this._hit) return;
      const rect = c.getBoundingClientRect(), mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const { P, order, rads } = this._hit;
      let hit = -1;
      for (let idx = order.length - 1; idx >= 0; idx--) {
        const a = order[idx];
        if (Math.hypot(mx - P[a][0], my - P[a][1]) <= rads[a] + 4) { hit = a; break; }
      }
      if (hit !== this.hoverAtom) { this.hoverAtom = hit; this.updateInspect(); this.schedule(); }
    });
    c.addEventListener('pointerup', () => { drag = null; });
    const clearHover = () => { drag = null; if (this.hoverAtom !== -1) { this.hoverAtom = -1; this.updateInspect(); this.schedule(); } };
    c.addEventListener('pointerleave', clearHover);
    c.addEventListener('pointercancel', clearHover);
    c.addEventListener('wheel', e => { e.preventDefault(); this.zoom = Math.max(.25, Math.min(8, this.zoom * Math.exp(-e.deltaY * .0012))); this.schedule(); }, { passive: false });
    c.addEventListener('keydown', e => {
      const k = { ArrowLeft: [-.12, 0], ArrowRight: [.12, 0], ArrowUp: [0, -.12], ArrowDown: [0, .12] }[e.key];
      if (k) { e.preventDefault(); this.yaw += k[0]; this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + k[1])); this.schedule(); }
      else if (e.key === '+' || e.key === '=') { this.zoom *= 1.15; this.schedule(); }
      else if (e.key === '-') { this.zoom /= 1.15; this.schedule(); }
      else if (this.frames.length > 1 && (e.key === ',' || e.key === '.')) { e.preventDefault(); this.step(e.key === ',' ? -1 : 1); }
    });
  }
  step(d) { const n = this.frames.length; this.f = (this.f + d + n) % n; if (this.slider) this.slider.value = this.f; this._last = performance.now(); this.updateInspect(); this.schedule(); }
  toggle() {
    this.playing = !this.playing;
    this.playBtn.textContent = this.playing ? 'Pause' : 'Play'; this.playBtn.setAttribute('aria-label', this.playing ? 'Pause' : 'Play');
    this._last = performance.now(); if (this.playing) this.animate();
  }
  stop() { this.playing = false; this.mode = -1; }
  animate() {
    if (this._raf) return;
    const tick = t => {
      this._raf = null;
      if (!this.root.isConnected) return;
      if (this.playing && this.frames.length > 1 && t - (this._last || 0) >= this.interval) {
        this._last = t; this.f = (this.f + 1) % this.frames.length;
        if (this.slider) this.slider.value = this.f;
        this.updateInspect();
      }
      if (this.mode >= 0) this.phase = t / 1000 * Math.PI * 2 * 0.9;
      this.draw();
      if (this.playing || this.mode >= 0) this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }
  schedule() { if (this._raf || this._pend) return; this._pend = true; requestAnimationFrame(() => { this._pend = false; this.draw(); }); }

  basePositions() {
    const fr = this.frames[this.f];
    if (this.mode >= 0 && this.modes[this.mode]) {
      const d = this.modes[this.mode].disp, a = 0.4 * Math.sin(this.phase);
      return fr.positions.map((p, i) => [p[0] + a * d[i][0], p[1] + a * d[i][1], p[2] + a * d[i][2]]);
    }
    return fr.positions;
  }

  palette(exportPng) {
    if (exportPng) return { text: '#15191e', muted: '#5a6470', force: '#c4561c', accent: '#1f4e8c', border: '#d9dde2', cardBg: 'rgba(255,255,255,0.94)' };
    const surf = cssVar('--surface') || '#ffffff', dark = surf.startsWith('#1') || surf.startsWith('#2');
    return {
      text: cssVar('--text') || '#15191e',
      muted: cssVar('--muted') || '#5a6470',
      force: cssVar('--force') || '#c4561c',
      accent: cssVar('--accent') || '#1f4e8c',
      border: cssVar('--border-strong') || '#b9c0c8',
      cardBg: dark ? 'rgba(25,28,31,0.88)' : 'rgba(255,255,255,0.88)',
    };
  }

  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    this.renderScene(g, w, h, dpr, { exportPng: false });
  }

  renderScene(g, w, h, dpr, opts = {}) {
    const fr = this.frames[this.f];
    if (!fr) { if (!opts.exportPng) this.label.textContent = ''; return; }
    const pal = this.palette(!!opts.exportPng);
    const activeCell = fr.cell || this.cell;
    const r = this.rep, cellR = scaleCell(activeCell, r), nb = fr.symbols.length;
    const sym = r > 1 ? expand(fr.symbols.map(s => [s]), activeCell, r, false).map(x => x[0]) : fr.symbols;
    const pos = expand(this.basePositions(), activeCell, r), n = sym.length;
    const lblText = (fr.label || '') + (this.frames.length > 1 ? `  ·  ${this.f + 1} of ${this.frames.length}` : '');
    if (!opts.exportPng) {
      this.label.textContent = lblText;
      if (this.count) this.count.textContent = `${this.f + 1} / ${this.frames.length}`;
    }
    if (!this._fit) this._fit = this.fit(expand(this.frames[0].positions, this.frames[0].cell || this.cell, r), cellR);
    const { c: ctr, r: R } = this._fit, scale = Math.min(w, h) * .46 / (R + 1) * this.zoom;
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch), D = R * 6 + 20;
    const proj = p => {
      const x = p[0] - ctr[0], y = p[1] - ctr[1], z = p[2] - ctr[2];
      const x1 = cy * x + sy * z, z1 = -sy * x + cy * z, y1 = cp * y - sp * z1, z2 = sp * y + cp * z1, k = 1 / (1 - z2 / D);
      return [w / 2 + x1 * scale * k, h / 2 - y1 * scale * k, z2, k];
    };
    const P = pos.map(proj);
    if (cellR) this.drawCell(g, cellR, proj, pal);
    const bkey = this.mode >= 0 ? 'm' + r : this.f + ':' + r;
    let bonds = this.bonds.get(bkey);
    if (!bonds) { bonds = bondsOf(sym, this.mode >= 0 ? expand(fr.positions, activeCell, r) : pos); this.bonds.set(bkey, bonds); }
    const big = n > 1500, atomScale = n > 300 ? .5 : .62;
    const rads = new Float32Array(n);
    for (let a = 0; a < n; a++) rads[a] = Math.max(1.5, (0.3 + 0.32 * radius(sym[a])) * scale * atomScale * P[a][3]);
    const order = new Array(n); for (let a = 0; a < n; a++) order[a] = a;
    order.sort((a, b) => P[a][2] - P[b][2]);
    if (!opts.exportPng) this._hit = { P, order, sym, pos, rads };

    const scalars = this.colorMode !== 'element' ? this.frameScalars(fr, this.colorMode) : null;
    const rng = scalars ? this.colorRanges[this.colorMode] : null;
    const atomHex = scalars && rng ? a => scalarColor(this.colorMode, scalars[a % nb], rng.lo, rng.hi) : a => color(sym[a]);

    const byAtom = new Map(); bonds.forEach(bd => { const lo = P[bd[0]][2] < P[bd[1]][2] ? bd[0] : bd[1]; let l = byAtom.get(lo); if (!l) byAtom.set(lo, l = []); l.push(bd); });
    g.lineCap = 'round'; g.lineWidth = Math.max(1.2, scale * .11);
    for (const a of order) {
      const bl = byAtom.get(a);
      if (bl) for (const [p, q] of bl) {
        const mx = (P[p][0] + P[q][0]) / 2, my = (P[p][1] + P[q][1]) / 2;
        g.strokeStyle = atomHex(p); g.beginPath(); g.moveTo(P[p][0], P[p][1]); g.lineTo(mx, my); g.stroke();
        g.strokeStyle = atomHex(q); g.beginPath(); g.moveTo(mx, my); g.lineTo(P[q][0], P[q][1]); g.stroke();
      }
      const rr = rads[a];
      const sp2 = scalars && rng ? spriteColor(atomHex(a), rr, big, dpr) : sprite(sym[a], rr, big, dpr);
      g.drawImage(sp2, P[a][0] - sp2.r - 1, P[a][1] - sp2.r - 1, 2 * sp2.r + 2, 2 * sp2.r + 2);
    }
    if (!opts.exportPng && this.hoverAtom >= 0 && this.hoverAtom < n) {
      const ha = this.hoverAtom, hr = rads[ha] + 3.5;
      g.save();
      g.beginPath(); g.arc(P[ha][0], P[ha][1], hr, 0, Math.PI * 2);
      g.lineWidth = 3.4; g.strokeStyle = 'rgba(255,255,255,0.92)'; g.stroke();
      g.lineWidth = 1.8; g.strokeStyle = pal.accent; g.stroke();
      g.restore();
    }
    if (this.showLabels && n <= 400) {
      g.fillStyle = pal.text; g.font = '11px ui-monospace, Menlo, monospace';
      for (const a of order) g.fillText(sym[a] + ((a % nb) + 1), P[a][0] + rads[a] * .7, P[a][1] - rads[a] * .7);
    }
    if (this.showForces && fr.forces) this.drawForces(g, fr, pos, P, proj, w, h, pal);
    if (this.showDipole && fr.dipole && !activeCell) this.drawDipole(g, fr, pos, proj, pal);
    if (scalars && rng) this.drawColorbar(g, h, rng, pal);
    else if (opts.exportPng) this.drawExportLegend(g, fr, h, pal);
    if (opts.exportPng && lblText) {
      g.fillStyle = pal.muted; g.font = '12px system-ui, -apple-system, sans-serif';
      g.fillText(lblText, 12, 20);
    }
  }

  drawColorbar(g, h, rng, pal) {
    const cm = COLORMAPS[this.colorMode];
    if (!cm) return;
    const cx = 10, cw = 188, ch = 44, cy = h - ch - 8, bx = cx + 8, by = cy + 17, bw = cw - 16, bh = 8;
    g.save();
    rrect(g, cx, cy, cw, ch, 5);
    g.fillStyle = pal.cardBg; g.fill();
    g.lineWidth = 1; g.strokeStyle = pal.border; g.stroke();

    g.fillStyle = pal.text; g.font = '600 10.5px system-ui, -apple-system, sans-serif';
    g.fillText(cm.label, bx, cy + 12);

    const grd = g.createLinearGradient(bx, 0, bx + bw, 0);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) grd.addColorStop(t, scalarColor(this.colorMode, rng.lo + t * (rng.hi - rng.lo), rng.lo, rng.hi));
    g.fillStyle = grd; g.fillRect(bx, by, bw, bh);
    g.strokeStyle = 'rgba(0,0,0,0.25)'; g.lineWidth = 0.8; g.strokeRect(bx, by, bw, bh);
    if (rng.diverging) {
      g.beginPath(); g.moveTo(bx + bw / 2, by - 1); g.lineTo(bx + bw / 2, by + bh + 1);
      g.strokeStyle = 'rgba(0,0,0,0.45)'; g.lineWidth = 1; g.stroke();
    }
    g.fillStyle = pal.muted; g.font = '10px ui-monospace, Menlo, monospace';
    g.textAlign = 'left'; g.fillText(fmtBarVal(rng.lo, rng.diverging), bx, cy + 38);
    g.textAlign = 'center'; g.fillText(fmtBarVal(rng.mid, false), bx + bw / 2, cy + 38);
    g.textAlign = 'right'; g.fillText(fmtBarVal(rng.hi, rng.diverging), bx + bw, cy + 38);
    g.restore();
  }

  drawExportLegend(g, fr, h, pal) {
    const syms = [...new Set(fr.symbols)];
    if (!syms.length || syms.length > 12) return;
    g.save();
    g.font = '11px system-ui, -apple-system, sans-serif';
    let x = 14; const y = h - 12;
    for (const s of syms) {
      g.beginPath(); g.arc(x + 4.5, y - 3.5, 4.5, 0, Math.PI * 2);
      g.fillStyle = color(s); g.fill();
      g.lineWidth = 0.8; g.strokeStyle = 'rgba(0,0,0,0.28)'; g.stroke();
      g.fillStyle = pal.muted; g.fillText(s, x + 12, y);
      x += 18 + g.measureText(s).width + 8;
    }
    g.restore();
  }

  exportPng() {
    if (!this.frames[this.f]) return;
    const w = Math.max(640, this.canvas.clientWidth || 800), h = Math.max(420, this.canvas.clientHeight || 540), dpr = 2;
    const out = document.createElement('canvas');
    out.width = Math.round(w * dpr); out.height = Math.round(h * dpr);
    const g = out.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#FFFFFF'; g.fillRect(0, 0, w, h);
    this.renderScene(g, w, h, dpr, { exportPng: true });
    const trigger = url => {
      const a = el('a', { href: url, download: 'mena-view.png' });
      document.body.append(a); a.click(); a.remove();
    };
    if (out.toBlob) {
      out.toBlob(blob => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        trigger(url);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }, 'image/png');
    } else trigger(out.toDataURL('image/png'));
  }

  fit(p0, cellR) {
    const c0 = [0, 1, 2].map(k => p0.reduce((s, p) => s + p[k], 0) / Math.max(1, p0.length));
    const ctr = cellR ? [0, 1, 2].map(k => (cellR[0][k] + cellR[1][k] + cellR[2][k]) / 2) : c0;
    let R = 1; p0.forEach(p => { R = Math.max(R, Math.hypot(p[0] - ctr[0], p[1] - ctr[1], p[2] - ctr[2])); });
    if (cellR) { const half = Math.hypot(...[0, 1, 2].map(k => (cellR[0][k] + cellR[1][k] + cellR[2][k]) / 2)); R = Math.max(R, Math.min(half, R * 1.6)); }
    return { c: ctr, r: R };
  }
  drawCell(g, v, proj, pal) {
    const P = []; for (let i = 0; i < 8; i++) P.push(proj([0, 1, 2].map(k => ((i & 1) ? v[0][k] : 0) + ((i & 2) ? v[1][k] : 0) + ((i & 4) ? v[2][k] : 0))));
    g.strokeStyle = pal.muted; g.globalAlpha = .6; g.lineWidth = 1; g.setLineDash([4, 3]); g.beginPath();
    for (let i = 0; i < 8; i++) for (const b of [1, 2, 4]) if (!(i & b)) { g.moveTo(P[i][0], P[i][1]); g.lineTo(P[i | b][0], P[i | b][1]); }
    g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
  }
  drawForces(g, fr, pos, P, proj, w, h, pal) {
    const F = expand(fr.forces, fr.cell || this.cell, this.rep, false);
    let fm = 1e-9;
    for (let i = 0; i < fr.forces.length; i++) { const f = fr.forces[i], m = Math.hypot(f[0], f[1], f[2]); if (m > fm) fm = m; }
    const L = 1.5, col = pal.force;
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2;
    F.forEach((f, a) => {
      const len = Math.hypot(...f) / fm; if (len < 0.02) return;
      const tip = proj([pos[a][0] + f[0] / fm * L, pos[a][1] + f[1] / fm * L, pos[a][2] + f[2] / fm * L]);
      g.beginPath(); g.moveTo(P[a][0], P[a][1]); g.lineTo(tip[0], tip[1]); g.stroke();
      const ang = Math.atan2(tip[1] - P[a][1], tip[0] - P[a][0]);
      g.beginPath(); g.moveTo(tip[0], tip[1]); g.lineTo(tip[0] - 7 * Math.cos(ang - .4), tip[1] - 7 * Math.sin(ang - .4)); g.lineTo(tip[0] - 7 * Math.cos(ang + .4), tip[1] - 7 * Math.sin(ang + .4)); g.fill();
    });
    g.fillStyle = pal.muted; g.font = '11px system-ui, sans-serif';
    g.textAlign = 'right'; g.fillText(`Arrows: forces, longest ${fmt(fm, 3)} eV/Å`, w - 10, h - 10); g.textAlign = 'left';
  }
  drawDipole(g, fr, pos, proj, pal) {
    const d = fr.dipole, dmag = Math.hypot(d[0], d[1], d[2]);
    if (dmag < 0.02 || !pos.length) return;
    const ctr = [0, 1, 2].map(k => pos.reduce((s, p) => s + p[k], 0) / pos.length);
    const L = Math.min(2.6, 0.8 + 0.45 * dmag), u = d.map(v => v / dmag);
    const p0 = proj([ctr[0] - 0.35 * L * u[0], ctr[1] - 0.35 * L * u[1], ctr[2] - 0.35 * L * u[2]]);
    const tip = proj([ctr[0] + 0.65 * L * u[0], ctr[1] + 0.65 * L * u[1], ctr[2] + 0.65 * L * u[2]]);
    const col = pal.accent;
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2.4;
    g.beginPath(); g.moveTo(p0[0], p0[1]); g.lineTo(tip[0], tip[1]); g.stroke();
    const ang = Math.atan2(tip[1] - p0[1], tip[0] - p0[0]);
    g.beginPath(); g.moveTo(tip[0], tip[1]);
    g.lineTo(tip[0] - 9 * Math.cos(ang - .42), tip[1] - 9 * Math.sin(ang - .42));
    g.lineTo(tip[0] - 9 * Math.cos(ang + .42), tip[1] - 9 * Math.sin(ang + .42));
    g.fill();
    g.font = '11px ui-monospace, Menlo, monospace';
    g.fillText(`μ = ${fmt(dmag, 2)} D`, tip[0] + 6, tip[1] - 6);
  }
  destroy() { if (this.ro) this.ro.disconnect(); this.stop(); this.root.remove(); }
}
