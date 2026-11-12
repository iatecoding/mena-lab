// Small helpers: DOM builder, number formatting, API calls, toasts, downloads, per-user storage.

export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(e.dataset, v);
    else if (v === true) e.setAttribute(k, '');
    else e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return e;
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

export function fmt(v, digits) {
  if (v === null || v === undefined || v === '') return '–';
  if (typeof v === 'string') return v;
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (!Number.isFinite(v)) return '–';
  if (digits !== null && digits !== undefined) {
    if (digits === 0) return Math.round(v).toLocaleString('en-US');
    if (Math.abs(v) >= 1e6) return v.toExponential(Math.min(digits, 3));
    if (Math.abs(v) < 10 ** -digits && v !== 0) return Math.abs(v) < 1e-4 ? v.toExponential(2) : v.toPrecision(2);
    return v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e6 || a < 1e-3) return v.toExponential(2);
  if (Number.isInteger(v)) return v.toLocaleString('en-US');
  return String(+v.toPrecision(5));
}

export function fmtDuration(s) {
  if (s === null || s === undefined || !Number.isFinite(s)) return '–';
  if (s < 0.1) return 'under 0.1 s';
  if (s < 10) return s.toFixed(1) + ' s';
  if (s < 60) return Math.round(s) + ' s';
  if (s < 3600) { const m = Math.floor(s / 60); return `${m} min ${String(Math.round(s % 60)).padStart(2, '0')} s`; }
  return `${Math.floor(s / 3600)} h ${String(Math.round((s % 3600) / 60)).padStart(2, '0')} min`;
}

export function fmtDate(ts) {
  if (!ts) return '–';
  const d = new Date(ts * 1000), now = new Date();
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return d.toDateString() === now.toDateString() ? time : d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + time;
}

export class ApiError extends Error {
  constructor(message, status, field, data) { super(message); this.status = status; this.field = field; this.data = data; }
}

// Every state-changing request carries X-Mena: 1 (the server refuses it otherwise).
export async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (method !== 'GET') { opts.headers['X-Mena'] = '1'; opts.headers['Content-Type'] = 'application/json'; if (body !== undefined) opts.body = JSON.stringify(body); }
  let r;
  try { r = await fetch(path, opts); } catch (e) { throw new ApiError('Cannot reach the Mena server. Is `mena webapp` still running?', 0); }
  let data = null;
  try { data = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new ApiError((data && data.error) || `Request failed (${r.status})`, r.status, data && data.field, data);
  return data;
}

export function toast(message, kind = 'info', ms = 5000) {
  const box = document.getElementById('toasts');
  const t = el('div', { class: 'toast ' + kind, role: kind === 'error' ? 'alert' : 'status' }, message);
  box.append(t);
  setTimeout(() => t.remove(), ms);
}

export function saveFile(name, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = el('a', { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied to the clipboard.'); }
  catch (e) {
    const ta = el('textarea', { style: 'position:fixed;left:-9999px' }); ta.value = text; document.body.append(ta); ta.select();
    try { document.execCommand('copy'); toast('Copied to the clipboard.'); } catch (e2) { toast('Copy failed: select the text and copy it by hand.', 'error'); }
    ta.remove();
  }
}

// Per-user conveniences only (theme, last task). Storage can be blocked: the app works without it.
export function store(key, value) {
  try {
    if (value === undefined) { const v = localStorage.getItem('mena-' + key); return v === null ? null : JSON.parse(v); }
    if (value === null) localStorage.removeItem('mena-' + key);
    else localStorage.setItem('mena-' + key, JSON.stringify(value));
  } catch (e) { /* private window or blocked storage */ }
  return null;
}

export function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)'];

export function approx(s) { return s != null && s < 0.1 ? 'under 0.1 s' : 'about ' + fmtDuration(s); }
