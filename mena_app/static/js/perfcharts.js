// Static SVG charts for the Performance view: horizontal bars on a log axis and a labelled scatter.
// Mena is drawn in the accent colour, every other model in neutral grey; each chart redraws on resize.
import { el, clear, fmt } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const s = (tag, attrs = {}, text) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; return e; };
const decades = (lo, hi) => { const out = []; for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) out.push(10 ** e); return out; };
const tick = v => v >= 1000 ? (v / 1000) + 'k' : String(+v.toPrecision(3));
const textW = (t, px = 11.5) => String(t).length * px * 0.56;

class Resizing {
  constructor(host, draw) {
    this.host = el('div', { class: 'pchart' }); host.append(this.host); this.draw = draw; this.w = 0;
    this.ro = new ResizeObserver(e => { const w = Math.round(e[0].contentRect.width); if (w > 120 && Math.abs(w - this.w) > 4) { this.w = w; clear(this.host).append(this.draw(w)); } });
    this.ro.observe(this.host);
  }
}

// rows: [{label, value, mena, title}], unit: "GB", linear?: bool, digits?: int
export function hbars(host, { rows, unit, x_label, linear = false, digits }) {
  return new Resizing(host, W => {
    const labW = Math.min(230, Math.max(...rows.map(r => textW(r.label, 11.5))) + 14), rowH = 22;
    const m = { l: labW, r: 82, t: 6, b: 34 }, H = m.t + m.b + rows.length * rowH, iw = W - m.l - m.r;
    const vals = rows.map(r => r.value);
    const lo = linear ? 0 : Math.min(...vals) / 1.6;
    const hi = Math.max(...vals) * (linear ? 1.18 : 1.25);
    const X = linear
      ? (v => m.l + (v / (hi || 1)) * iw)
      : (v => m.l + (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * iw);
    const fmtVal = v => digits != null ? fmt(v, digits) : fmt(v);
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${x_label}: ${rows.map(r => `${r.label} ${fmtVal(r.value)} ${unit}`).join('; ')}` });
    const grid = s('g', { class: 'grid' }), ax = s('g', { class: 'axis' });
    if (linear) {
      const rawStep = hi / 4;
      const mag = 10 ** Math.floor(Math.log10(rawStep || 1));
      const norm = rawStep / mag;
      const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
      for (let v = 0; v <= hi + 1e-9; v += step) {
        if (v > hi) break;
        grid.append(s('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }));
        ax.append(s('text', { x: X(v), y: H - m.b + 15, 'text-anchor': 'middle' }, tick(+v.toPrecision(6))));
      }
    } else {
      decades(lo, hi).filter(v => v >= lo && v <= hi).forEach(v => {
        grid.append(s('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }));
        ax.append(s('text', { x: X(v), y: H - m.b + 15, 'text-anchor': 'middle' }, tick(v)));
      });
    }
    ax.append(s('text', { x: m.l + iw / 2, y: H - 4, 'text-anchor': 'middle' }, x_label));
    svg.append(grid, ax);
    rows.forEach((r, i) => {
      const y = m.t + i * rowH, cls = r.mena ? 'bar mena' : 'bar';
      const g = s('g', { class: cls });
      g.append(s('title', {}, r.title || `${r.label}: ${fmtVal(r.value)} ${unit}`));
      g.append(s('text', { x: m.l - 8, y: y + rowH / 2 + 4, 'text-anchor': 'end', class: 'blabel' }, r.label));
      g.append(s('rect', { x: m.l, y: y + 4, width: Math.max(1, X(r.value) - m.l), height: rowH - 8, rx: 2 }));
      g.append(s('text', { x: X(r.value) + 6, y: y + rowH / 2 + 4, class: 'bval' }, `${fmtVal(r.value)} ${unit}`));
      svg.append(g);
    });
    return svg;
  });
}

// points: [{name, x, y, mena, title}]; log x. Labels are placed greedily (Mena first, then by y) and skipped if they would collide.
export function scatter(host, { points, x_label, y_label, x_unit = '' }) {
  return new Resizing(host, W => {
    const H = Math.round(Math.max(300, Math.min(W * 0.62, 420))), m = { l: 54, r: 22, t: 16, b: 42 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const xlo = Math.min(...xs) / 1.45, xhi = Math.max(...xs) * 1.45;
    const ypad = (Math.max(...ys) - Math.min(...ys)) * 0.08, ylo = Math.min(...ys) - ypad, yhi = Math.max(...ys) + ypad * 1.6;
    const X = v => m.l + (Math.log10(v) - Math.log10(xlo)) / (Math.log10(xhi) - Math.log10(xlo)) * iw, Y = v => m.t + (1 - (v - ylo) / (yhi - ylo)) * ih;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${y_label} against ${x_label} for ${points.length} models` });
    const grid = s('g', { class: 'grid' }), ax = s('g', { class: 'axis' });
    [1, 2, 5].flatMap(k => decades(xlo, xhi).map(d => k * d)).filter(v => v >= xlo && v <= xhi).sort((a, b) => a - b).forEach(v => {
      grid.append(s('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }));
      ax.append(s('text', { x: X(v), y: H - m.b + 15, 'text-anchor': 'middle' }, tick(v) + x_unit));
    });
    const step = (yhi - ylo) > 0.25 ? 0.1 : (yhi - ylo) > 0.12 ? 0.05 : 0.02;
    for (let v = Math.ceil(ylo / step) * step; v <= yhi; v += step) {
      grid.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      ax.append(s('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, v.toFixed(2)));
    }
    ax.append(s('text', { x: m.l + iw / 2, y: H - 4, 'text-anchor': 'middle' }, x_label),
      s('text', { x: 13, y: m.t + ih / 2, 'text-anchor': 'middle', transform: `rotate(-90 13 ${m.t + ih / 2})` }, y_label));
    svg.append(grid, ax);
    const dots = s('g'), labels = s('g'), placed = [];
    const order = [...points].sort((a, b) => (b.mena - a.mena) || (b.y - a.y));
    [...points].sort((a, b) => a.mena - b.mena).forEach(p => {
      const g = s('g', { class: p.mena ? 'dot mena' : 'dot' });
      g.append(s('title', {}, p.title || p.name), s('circle', { cx: X(p.x), cy: Y(p.y), r: p.mena ? 6.5 : 3.8 }));
      dots.append(g);
    });
    order.forEach(p => {
      const px = X(p.x), py = Y(p.y), fs = p.mena ? 12.5 : 10.2, w = textW(p.name, fs), h = fs + 2;
      const tries = [[px + 8, py + 4], [px - 8 - w, py + 4], [px - w / 2, py - 9], [px - w / 2, py + fs + 7]];
      for (const [tx, ty] of tries) {
        const box = { x: tx - 2, y: ty - h + 2, w: w + 4, h };
        if (box.x < m.l || box.x + box.w > W - m.r + 6 || box.y < 0 || box.y + box.h > H - m.b) continue;
        if (placed.some(q => box.x < q.x + q.w && q.x < box.x + box.w && box.y < q.y + q.h && q.y < box.y + box.h)) continue;
        if (points.some(q => q !== p && Math.abs(X(q.x) - (box.x + box.w / 2)) < box.w / 2 && Math.abs(Y(q.y) - (box.y + box.h / 2)) < box.h / 2 + 2)) continue;
        placed.push(box);
        labels.append(s('text', { x: tx, y: ty, class: p.mena ? 'dlabel mena' : 'dlabel' }, p.name));
        break;
      }
    });
    svg.append(dots, labels);
    return svg;
  });
}
