// Hand-written SVG charts: line (optional markers), bar, sticks. Live append, log axes, labelled x guides (band paths),
// single-colour mode for many series (legend: false), tooltip, and a "data" table toggle.
import { el, clear, fmt, SERIES } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const s = (tag, attrs = {}, text) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; return e; };

function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) hi = lo + 1;
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(v => v >= raw) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
function logTicks(lo, hi) {
  const a = Math.floor(Math.log10(lo)), b = Math.ceil(Math.log10(hi));
  for (const mults of [[1], [1, 3], [1, 2, 5]]) {
    const out = [];
    for (let e = a; e <= b; e++) for (const m of mults) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
    if (out.length >= 3) return out;
  }
  return niceTicks(lo, hi, 4);
}
// Band-path breaks give two labels at one x ("U" then "K"): show them once, as "U|K".
function mergeTicks(ticks, eps) {
  const out = [];
  [...ticks].sort((a, b) => a.x - b.x).forEach(t => {
    const last = out[out.length - 1];
    if (last && Math.abs(last.x - t.x) <= eps) { if (last.label !== t.label) last.label += '|' + t.label; }
    else out.push({ x: t.x, label: t.label });
  });
  return out;
}
const tickLabel = v => { const a = Math.abs(v); return a >= 1e5 || (a > 0 && a < 1e-3) ? v.toExponential(0) : String(+v.toPrecision(4)); };

export class Chart {
  constructor(container, spec) {
    this.spec = { ...spec };
    this.series = (spec.series || []).map(q => ({ name: q.name, x: [...(q.x || [])], y: [...(q.y || [])] }));
    this.showTable = false;
    this.tableBtn = el('button', { class: 'btn ghost small', type: 'button', 'aria-pressed': 'false', title: 'Show the numbers behind this chart',
      onclick: () => { this.showTable = !this.showTable; this.render(); } }, 'Data');
    this.root = el('figure', { class: 'chart' + (spec.xticks && spec.xticks.length ? ' wide' : '') });
    this.legend = el('div', { class: 'legend' });
    this.host = el('div', { class: 'chart-plot' });
    this.table = el('div', { class: 'data-table', hidden: true });
    this.tip = el('div', { class: 'tip', hidden: true });
    this.root.append(el('figcaption', { class: 'chart-top' }, el('span', { class: 'chart-title' }, spec.title || ''), this.tableBtn),
      this.legend, this.host, this.table, this.tip);
    container.append(this.root);
    this.W = 520; this.H = this.height(this.W); this.pending = false;
    this.ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(entries => {
      const w = Math.round(entries[0].contentRect.width);
      if (w >= 160 && Math.abs(w - this.W) > 6) { this.W = w; this.H = this.height(w); this.schedule(); }
    }) : null;
    if (this.ro) this.ro.observe(this.host);
    this.render();
  }
  // Compact charts sit stacked beside the viewer, so they are shorter.
  height(w) { return Math.round(this.spec.compact ? Math.max(150, Math.min(w * 0.5, 215)) : Math.max(200, Math.min(w * 0.52, 300))); }
  destroy() { if (this.ro) this.ro.disconnect(); this.root.remove(); }
  schedule() { if (this.pending) return; this.pending = true; requestAnimationFrame(() => { this.pending = false; this.render(); }); }
  append(name, xs, ys) {
    let q = this.series.find(v => v.name === name);
    if (!q) { q = { name, x: [], y: [] }; this.series.push(q); }
    q.x.push(...xs); q.y.push(...ys);
    this.schedule();
  }
  single() { return this.spec.legend === false; }
  color(i) {
    if (this.single()) return SERIES[0];
    const fixed = this.spec.seriesColors && this.series[i] && this.spec.seriesColors[this.series[i].name];
    return SERIES[(fixed ?? i) % SERIES.length];
  }

  render() {
    const sp = this.spec, kind = sp.kind || 'line', cat = kind === 'bar';
    const all = this.series.filter(q => q.x.length);
    clear(this.legend); clear(this.host);
    this.tableBtn.setAttribute('aria-pressed', String(this.showTable));
    this.legend.hidden = this.single() || this.series.length < 2;
    if (!this.legend.hidden) this.series.forEach((q, i) => this.legend.append(el('span', {}, el('i', { style: `background:${this.color(i)}` }), q.name)));
    if (!all.length) { this.host.append(el('div', { class: 'chart-empty' }, 'Waiting for data…')); this.renderTable(); return; }
    const W = this.W, H = this.H, xg = Array.isArray(sp.xticks) && sp.xticks.length;
    const logx = !!sp.logx && !cat;
    // A log axis cannot show zeros: fall back to linear when nothing is positive (e.g. a force that is exactly zero).
    const logy = !!sp.logy && all.some(q => q.y.some(v => v > 0));
    const xsNum = cat ? [] : all.flatMap(q => q.x).filter(v => v !== null && Number.isFinite(v) && (!logx || v > 0));
    const ys = all.flatMap(q => q.y).filter(v => v !== null && Number.isFinite(v) && (!logy || v > 0));
    if (!ys.length || (!cat && !xsNum.length)) { this.host.append(el('div', { class: 'chart-empty' }, 'No plottable values.')); return; }
    let ylo = Infinity, yhi = -Infinity, xlo, xhi, cats = [];
    for (let i = 0; i < ys.length; i++) { if (ys[i] < ylo) ylo = ys[i]; if (ys[i] > yhi) yhi = ys[i]; }
    if (cat) { cats = all[0].x; xlo = -0.5; xhi = cats.length - 0.5; ylo = Math.min(0, ylo); yhi = Math.max(0, yhi); }
    else {
      xlo = Infinity; xhi = -Infinity;
      for (let i = 0; i < xsNum.length; i++) { if (xsNum[i] < xlo) xlo = xsNum[i]; if (xsNum[i] > xhi) xhi = xsNum[i]; }
    }
    if (kind === 'sticks') { ylo = 0; yhi = 1.1; const w = xhi - xlo; xlo = Math.max(0, xlo - w * .05 - 50); xhi += w * .05 + 50; }
    if (xhi === xlo) { xlo -= 1; xhi += 1; }
    if (yhi === ylo) { const d = Math.abs(yhi) * .1 || 1; yhi += d; ylo -= d; }
    if (!logy && kind !== 'sticks') { const pad = (yhi - ylo) * .06; if (!(cat && ylo === 0)) ylo -= pad; yhi += pad; }
    if (logy) { ylo /= 1.3; yhi *= 1.3; }
    const yt = kind === 'sticks' ? [] : (logy ? logTicks(ylo, yhi) : niceTicks(ylo, yhi, 5));
    const labW = Math.max(0, ...yt.map(v => tickLabel(v).length)) * 6.6;
    const ml = Math.max(40, Math.round(labW + (sp.y_label ? 30 : 14))), mr = 14;
    const catMaxLen = cat ? cats.reduce((mx, c) => Math.max(mx, String(c).length), 0) : 0;
    const catSlotW = cat && cats.length ? (W - ml - mr) / cats.length : Infinity;
    const catStagger = cat && cats.length > 1 && catMaxLen * 6.4 > catSlotW - 6;
    const m = { l: ml, r: mr, t: 10, b: (sp.x_label ? 40 : 26) + (catStagger ? 13 : 0) };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const fx = logx ? v => (Math.log10(v) - Math.log10(xlo)) / (Math.log10(xhi) - Math.log10(xlo)) : v => (v - xlo) / (xhi - xlo);
    const fy = logy ? v => (Math.log10(v) - Math.log10(ylo)) / (Math.log10(yhi) - Math.log10(ylo)) : v => (v - ylo) / (yhi - ylo);
    const X = v => m.l + fx(v) * iw, Y = v => m.t + (1 - fy(v)) * ih;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${sp.title || 'Chart'}: ${all.length} series` });
    const grid = s('g', { class: 'grid' }), ax = s('g', { class: 'axis' });
    yt.forEach(v => { if (v < ylo || v > yhi) return; grid.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) })); ax.append(s('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, tickLabel(v))); });
    if (!logy && ylo < 0 && yhi > 0) grid.append(s('line', { class: 'zero', x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0) }));
    if (xg) {
      mergeTicks(sp.xticks, (xhi - xlo) * 1e-4).forEach(t => {
        const px = X(t.x); if (px < m.l - 1 || px > W - m.r + 1) return;
        grid.append(s('line', { class: 'guide', x1: px, x2: px, y1: m.t, y2: H - m.b }));
        ax.append(s('text', { x: px, y: H - m.b + 16, 'text-anchor': 'middle', class: 'xtick-label' }, t.label));
      });
    } else {
      const xt = cat ? cats.map((_, i) => i) : (logx ? logTicks(xlo, xhi) : niceTicks(xlo, xhi, Math.max(3, Math.floor(iw / 90))));
      const maxXL = Math.max(1, Math.floor(iw / (cat ? 28 : 44)));
      xt.forEach((v, i) => {
        if (cat && cats.length > maxXL && i % Math.ceil(cats.length / maxXL)) return;
        const px = X(v); if (px < m.l - 1 || px > W - m.r + 1) return;
        const dy = catStagger && (i % 2 === 1) ? 12 : 0;
        ax.append(s('line', { x1: px, x2: px, y1: H - m.b, y2: H - m.b + 4 + dy }), s('text', { x: px, y: H - m.b + 16 + dy, 'text-anchor': 'middle' }, cat ? String(cats[v]) : tickLabel(v)));
      });
    }
    ax.append(s('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }), s('line', { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b }));
    if (sp.x_label) ax.append(s('text', { x: m.l + iw / 2, y: H - 6, 'text-anchor': 'middle' }, sp.x_label));
    if (sp.y_label) ax.append(s('text', { x: 13, y: m.t + ih / 2, 'text-anchor': 'middle', transform: `rotate(-90 13 ${m.t + ih / 2})` }, sp.y_label));
    svg.append(grid, ax);
    const marks = s('g');
    if (cat) {
      const bw = Math.max(2, Math.min(28, iw / cats.length * .7 / all.length));
      all.forEach((q, si) => q.y.forEach((v, i) => {
        if (v === null) return; const cx = X(i) + (si - (all.length - 1) / 2) * (bw + 2), y0 = Y(Math.max(ylo, 0)), y1 = Y(v);
        marks.append(s('rect', { x: cx - bw / 2, width: bw, y: Math.min(y0, y1), height: Math.max(1, Math.abs(y0 - y1)), fill: this.color(this.series.indexOf(q)) }));
      }));
    } else if (kind === 'sticks') {
      all.forEach(q => q.x.forEach((v, i) => marks.append(s('line', { x1: X(v), x2: X(v), y1: Y(0), y2: Y(q.y[i] ?? 1), stroke: this.color(0), 'stroke-width': 1.6 }))));
    } else {
      const sw = this.single() ? 1.5 : 1.8;
      all.forEach(q => {
        const si = this.series.indexOf(q), pts = [];
        q.x.forEach((xv, i) => { const yv = q.y[i]; if (yv === null || !Number.isFinite(yv) || (logy && yv <= 0) || (logx && xv <= 0)) return; pts.push([X(xv), Y(yv)]); });
        if (!pts.length) return;
        marks.append(s('path', { d: pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' '), fill: 'none', stroke: this.color(si), 'stroke-width': sw, 'stroke-linejoin': 'round' }));
        if (sp.markers || (!this.single() && pts.length <= 20)) pts.forEach(p => marks.append(s('circle', { cx: p[0], cy: p[1], r: 3.2, fill: this.color(si), stroke: 'var(--surface)', 'stroke-width': 1.5 })));
      });
    }
    svg.append(marks);
    this.hover(svg, all, { X, Y, m, W, H, iw, ih, cat, cats, logx });
    this.host.append(svg);
    this.renderTable();
  }

  hover(svg, all, g) {
    const cross = s('line', { y1: g.m.t, y2: g.H - g.m.b, class: 'cross', visibility: 'hidden' });
    const dots = s('g'), overlay = s('rect', { x: g.m.l, y: g.m.t, width: g.iw, height: g.ih, fill: 'transparent' });
    svg.append(cross, dots, overlay);
    const sp = this.spec;
    overlay.addEventListener('pointermove', ev => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (g.W / r.width), py = (ev.clientY - r.top) * (g.H / r.height);
      let best = null;
      all.forEach(q => q.x.forEach((xv, i) => {
        const yv = q.y[i]; if (yv === null || !Number.isFinite(yv)) return;
        const cx = g.cat ? g.X(i) : (g.logx && xv <= 0 ? -1e9 : g.X(xv));
        const d = this.single() ? Math.hypot(cx - px, g.Y(yv) - py) : Math.abs(cx - px);
        if (!best || d < best.d) best = { d, i, xv, cx, q };
      }));
      if (!best) return;
      clear(dots); clear(this.tip);
      cross.setAttribute('x1', best.cx); cross.setAttribute('x2', best.cx); cross.setAttribute('visibility', 'visible');
      const xText = g.cat ? g.cats[best.i] : (sp.xticks ? '' : fmt(best.xv));
      if (xText !== '') this.tip.append(el('div', { class: 'tip-x' }, `${sp.x_label || 'x'} = ${xText}`));
      const show = this.single() ? [best.q] : all;
      show.forEach(q => {
        const j = g.cat || q === best.q ? best.i : q.x.reduce((bj, xv, k) => (Math.abs(xv - best.xv) < Math.abs(q.x[bj] - best.xv) ? k : bj), 0);
        if (q.y[j] === undefined || q.y[j] === null) return;
        const si = this.series.indexOf(q);
        this.tip.append(el('div', {}, el('i', { style: `background:${this.color(si)}` }), show.length > 1 || this.single() ? q.name + ': ' : '', el('b', {}, fmt(q.y[j]))));
        dots.append(s('circle', { cx: g.cat ? g.X(j) : g.X(q.x[j]), cy: g.Y(q.y[j]), r: 4.5, fill: this.color(si), stroke: 'var(--surface)', 'stroke-width': 2 }));
      });
      this.tip.hidden = false;
      const hr = this.root.getBoundingClientRect(), tw = this.tip.offsetWidth;
      this.tip.style.left = Math.max(4, Math.min((ev.clientX - hr.left) + 14, hr.width - tw - 4)) + 'px';
      this.tip.style.top = (ev.clientY - hr.top - 10) + 'px';
    });
    overlay.addEventListener('pointerleave', () => { this.tip.hidden = true; cross.setAttribute('visibility', 'hidden'); clear(dots); });
  }

  renderTable() {
    this.table.hidden = !this.showTable; clear(this.table);
    if (!this.showTable) return;
    const t = el('table', { class: 'tbl' }), body = el('tbody');
    const cols = this.series.slice(0, 40);
    t.append(el('thead', {}, el('tr', {}, el('th', { text: this.spec.x_label || 'x' }), ...cols.map(q => el('th', { text: q.name })))));
    const n = Math.max(...cols.map(q => q.x.length));
    const same = cols.every(q => q.x.length === n && q.x.every((v, i) => v === cols[0].x[i]));
    if (same) for (let i = 0; i < Math.min(n, 500); i++) body.append(el('tr', {}, el('td', { text: fmt(cols[0].x[i]) }), ...cols.map(q => el('td', { text: fmt(q.y[i]) }))));
    else {
      const xs = [...new Set(cols.flatMap(q => q.x))].slice(0, 500);
      xs.forEach(x => body.append(el('tr', {}, el('td', { text: fmt(x) }), ...cols.map(q => { const i = q.x.indexOf(x); return el('td', { text: i < 0 ? '' : fmt(q.y[i]) }); }))));
    }
    t.append(body); this.table.append(t);
  }
}
