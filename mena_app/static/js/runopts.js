// Runtime overrides ("Tune") and job code export ("CLI / Python / JSON") inside the dock drawer.
import { el, clear, copyText, saveFile } from './util.js';
import { state, emit, on, domain } from './state.js';
import { lotLabel, deviceLabel } from './labels.js';

let optsHost, reproHost, tab = 'cli';

function autoModel(info, dom) {
  const avail = (info.models || []).filter(m => (m.available || m.downloadable) && (!dom || (m.domains || []).includes(dom)));
  if (!avail.length) return null;
  const pref = dom && info.settings && info.settings.default_model[dom];
  const mine = pref && avail.find(m => m.id === pref);
  if (mine) return mine;
  const rankOf = m => (m.rank && dom && m.rank[dom] !== undefined) ? m.rank[dom] : 99;
  return avail.slice().sort((a, b) => rankOf(a) - rankOf(b))[0];
}

function choices(p) {
  const info = state.info, auto = ['auto', 'Automatic'];
  if (p.name === 'model') return [auto, ...info.models.map(m => [m.id, m.title + (m.available ? '' : m.downloadable ? ' (downloads on first use)' : ' (not installed)'), !(m.available || m.downloadable)])];
  if (p.name === 'device') return [auto, ...info.machine.devices.map(d => [d, deviceLabel(d)])];
  if (p.name === 'lot') {
    const id = state.runtime.model || (state.plan && state.plan.decision.model);
    const m = info.models.find(x => x.id === id) || autoModel(info, domain());
    return [auto, ...((m && m.lots) || []).map(l => [l, lotLabel(l)])];
  }
  const names = { auto: 'Automatic', on: 'On', off: 'Off', float32: 'Single (float32)', float64: 'Double (float64)' };
  return (p.options || []).map(o => [o, names[o] || o]);
}

function renderOpts() {
  clear(optsHost);
  let pruned = false;
  const grid = el('div', { class: 'runtime-grid' });
  state.info.catalog.runtime.forEach(p => {
    const id = 'rt-' + p.name, cur = state.runtime[p.name];
    let ctl;
    if (p.type === 'choice') {
      const opts = choices(p);
      if (cur !== undefined && !opts.some(o => o[0] === cur && !o[2])) { delete state.runtime[p.name]; pruned = true; }
      ctl = el('select', { id, dataset: { field: p.name }, title: p.help || null, onchange: () => {
        if (ctl.value === 'auto') delete state.runtime[p.name]; else state.runtime[p.name] = ctl.value;
        if (p.name === 'model') { delete state.runtime.lot; renderOpts(); }
        emit('changed');
      } }, ...opts.map(([v, t, dis]) => el('option', { value: v, selected: v === (state.runtime[p.name] ?? 'auto'), disabled: dis || null }, t)));
    } else {
      ctl = el('input', { id, type: 'number', class: 'num', min: p.min, max: p.max, step: 1, value: cur ?? '', placeholder: 'auto',
        dataset: { field: p.name }, title: p.help || null,
        oninput: () => { const t = ctl.value.trim(); if (t === '') delete state.runtime[p.name]; else state.runtime[p.name] = Number(t); emit('changed'); } });
    }
    grid.append(el('div', { class: 'field' },
      el('div', { class: 'field-head' }, el('label', { for: id }, p.label),
        p.help ? el('span', { class: 'hint-dot', title: p.help, 'aria-label': p.help }, '?') : null),
      ctl));
  });
  const n = Object.keys(state.runtime).length;
  optsHost.append(grid, el('div', { style: 'margin-top:.45rem;display:flex;justify-content:flex-end' },
    el('button', { class: 'btn ghost small', type: 'button', disabled: !n || null, onclick: () => { state.runtime = {}; renderOpts(); emit('changed'); } }, 'Reset to automatic')));
  if (pruned && state.planError && !state.planPending) setTimeout(() => emit('changed'), 0);
}

const TABS = [['cli', 'CLI'], ['python', 'Python'], ['json', 'job.json']];

function renderRepro() {
  clear(reproHost);
  const ex = state.plan && state.plan.exports;
  if (!ex) { reproHost.append(el('p', { class: 'muted', style: 'margin:0' }, 'Available once the plan is ready.')); return; }
  const bar = el('div', { class: 'tabs-mini', role: 'tablist', 'aria-label': 'Export format' },
    ...TABS.map(([k, t]) => el('button', { type: 'button', role: 'tab', 'aria-selected': String(k === tab), onclick: () => { tab = k; renderRepro(); } }, t)));
  const text = ex[tab] || '';
  const name = tab === 'json' ? 'job.json' : tab === 'python' ? 'mena_job.py' : 'mena_job.sh';
  reproHost.append(bar, el('pre', { class: 'code', tabindex: 0, role: 'tabpanel' }, text),
    el('div', { class: 'inline' },
      el('button', { class: 'btn small', type: 'button', onclick: () => copyText(text) }, 'Copy'),
      el('button', { class: 'btn small', type: 'button', onclick: () => saveFile(name, text, tab === 'json' ? 'application/json' : 'text/plain') }, `Download ${name}`)));
}

export function mountRunOpts(oh, rh) {
  optsHost = oh; reproHost = rh;
  renderOpts(); renderRepro();
  on('plan', () => { renderRepro(); if (!state.runtime.model) renderOpts(); });
  on('restore-runtime', renderOpts);
}
