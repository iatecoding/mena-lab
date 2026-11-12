// Generic report renderer: puts headline, downloads & metric tiles at the top of the stage (opts.topHost when given)
// so results are immediately visible without scrolling, and renders charts + collapsible CSV-exportable tables below.
import { el, clear, fmt } from './util.js';
import { Chart } from './charts.js';

function conciseSub(hint) {
  if (!hint) return null;
  const raw = String(hint).split(' · ')[0].trim();
  if (/^Vector:\s*/.test(raw)) return raw.replace(/^Vector:\s*/, '');
  if (/^Anisotropy\s*/.test(raw)) return raw.replace(/^Anisotropy\s*/, '');
  const s = raw.split(' (')[0].trim();
  if (/^[+-−]?\d/.test(s)) return s;
  return raw.length <= 22 ? raw : null;
}

function tile(t) {
  const has = t.value !== null && t.value !== undefined;
  const flag = { pass: 'OK', warn: 'CHECK', fail: 'FAIL' }[t.status];
  const sub = conciseSub(t.hint);
  return el('div', { class: 'tile ' + (t.status || ''), title: t.hint || null },
    flag ? el('span', { class: 'tile-flag' }, flag) : null,
    el('div', { class: 'tile-k' }, t.label),
    el('div', { class: 'tile-v num' }, has ? fmt(t.value, t.digits) : '–', t.unit && has ? el('span', { class: 'tile-u' }, t.unit) : null),
    sub ? el('div', { class: 'tile-h' }, sub) : null);
}

function downloadCsv(t) {
  const esc = v => {
    const s = v === null || v === undefined ? '' : String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [t.columns.map(esc).join(','), ...t.rows.map(r => r.map(esc).join(','))].join('\n') + '\n';
  const slug = (t.title || 'table').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'table';
  const url = URL.createObjectURL(new Blob([lines], { type: 'text/csv;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url; a.download = `mena_${slug}.csv`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function table(t, onRow, openByDefault = false) {
  const tbl = el('table', { class: 'tbl' });
  tbl.append(el('thead', {}, el('tr', {}, t.columns.map(c => el('th', { scope: 'col', text: c })))));
  const body = el('tbody');
  t.rows.forEach((r, i) => {
    const tr = el('tr', {}, r.map(c => el('td', { class: typeof c === 'number' ? 'num' : '', text: c === null || c === undefined ? '–' : (typeof c === 'number' ? fmt(c) : String(c)) })));
    if (onRow) { tr.classList.add('clickable'); tr.tabIndex = 0; tr.addEventListener('click', () => onRow(i)); tr.addEventListener('keydown', e => { if (e.key === 'Enter') onRow(i); }); }
    body.append(tr);
  });
  tbl.append(body);
  const csvBtn = el('button', {
    class: 'btn small csv-btn', type: 'button', title: `Download ${t.title} as CSV`,
    onclick: e => { e.preventDefault(); e.stopPropagation(); downloadCsv(t); },
  }, '⬇ CSV');
  return el('details', { class: 'more table-box', open: openByDefault ? '' : null },
    el('summary', {}, t.title, el('span', { class: 'count' }, `${t.rows.length} rows`), csvBtn),
    el('div', { class: 'table-wrap' }, tbl), t.note ? el('p', { class: 'note' }, t.note) : null);
}

function downloadsBar(job, report, onClear) {
  const wrap = el('div', { class: 'files' });
  if (job) {
    const files = (job.files && job.files.length ? job.files : report.files || []);
    files.forEach(f => wrap.append(el('a', { class: 'btn small', href: `/api/jobs/${job.id}/files/${encodeURIComponent(f.name)}`, download: '' }, '⬇ ' + (f.label || f.name))));
    wrap.append(el('a', { class: 'btn small', href: `/api/jobs/${job.id}/files/result.json`, download: '' }, '⬇ JSON'));
  }
  if (onClear) wrap.append(el('button', { class: 'btn ghost small', type: 'button', title: 'Dismiss result and expand 3D viewer', onclick: onClear }, '× Clear'));
  return wrap;
}

// opts: {job, viewer, topHost, chartHost, onClear}
export function renderReport(host, report, opts = {}) {
  clear(host);
  const top = opts.topHost || host;
  if (opts.topHost) { clear(top); top.hidden = false; }
  const parts = [], job = opts.job;

  const titleWrap = el('div', { class: 'res-title-wrap' },
    el('div', { class: 'res-kicker' },
      el('span', { class: 'status ' + (report.cancelled ? 'cancelled' : 'done') }, report.cancelled ? 'Stopped early' : 'Completed'),
      job && job.duration_s != null ? el('span', { class: 'muted num', style: 'font-size:.78rem' }, `${fmt(job.duration_s, 2)} s`) : null),
    report.headline ? el('p', { class: 'headline' }, report.headline) : null);

  top.append(el('div', { class: 'res-bar' }, titleWrap, downloadsBar(job, report, opts.onClear)));
  if (report.cancelled) top.append(el('div', { class: 'warnbox' }, 'This run was stopped before finishing. Numbers below cover completed steps.'));
  if (report.tiles && report.tiles.length) top.append(el('div', { class: 'tiles' }, report.tiles.map(tile)));

  const shown = (report.charts || []).filter(c => c.series && c.series.some(q => q.x && q.x.length));
  if (shown.length && opts.chartHost) {
    clear(opts.chartHost); opts.chartHost.hidden = false;
    const g = el('div', { class: 'chart-group' }); opts.chartHost.append(g);
    shown.forEach(c => parts.push(new Chart(g, { ...c, compact: opts.compact !== false, ...(opts.chartMeta || {}) })));
  } else if (shown.length) {
    const grid = el('div', { class: 'charts' }); host.append(grid);
    shown.forEach(c => parts.push(new Chart(grid, c)));
  }

  const modes = report.viewer && report.viewer.modes;
  const tables = report.tables || [];
  // Open compact physical summary tables (e.g. energy decomposition or small frequency tables) automatically; keep per-atom tables folded.
  const autoOpenIdx = tables.findIndex(t => t.rows && t.rows.length > 0 && t.rows.length <= 20 && !/per-atom/i.test(t.title || ''));
  tables.forEach((t, idx) => {
    const freqTable = modes && modes.length && t.rows.length === modes.length && opts.viewer;
    host.append(table(t, freqTable ? i => opts.viewer.setMode(i) : null, idx === autoOpenIdx));
  });

  const facts = Object.entries(report.facts || {});
  if (facts.length) host.append(el('dl', { class: 'facts', style: tables.length ? 'margin-top:.55rem;margin-bottom:0' : '' }, facts.flatMap(([k, v]) => [el('dt', {}, k), el('dd', { class: typeof v === 'number' ? 'num' : '' }, fmt(v))])));

  if (report.notes && report.notes.length) host.append(el('ul', { class: 'notes' }, report.notes.map(n => el('li', {}, n))));
  if (opts.topHost) host.hidden = host.childElementCount === 0;

  return {
    charts: shown.length,
    destroy() {
      parts.forEach(p => p.destroy());
      if (opts.topHost) { clear(top); top.hidden = true; }
      if (opts.chartHost && shown.length) { clear(opts.chartHost); opts.chartHost.hidden = true; }
    },
  };
}
