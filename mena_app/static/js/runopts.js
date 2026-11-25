// Runtime overrides ("Tune") and job code export ("CLI / Python / JSON") inside the dock drawer.
import { el, clear, copyText, saveFile, toast } from './util.js';
import { state, emit, on, domain } from './state.js';
import { lotLabel, deviceLabel, PRECISION } from './labels.js';

let optsHost, reproHost, tab = 'cli';

const MOL_LOTS = new Set([
  'omol_wb97mv', 'opoly26_wb97mv', 'openrxn_wb97xd', 'spice_wb97md3',
  'qcml_pbe0', 't1x_wb97x', 'wiggle_wb97mv', 'nabladft_wb97xd',
]);

function autoModel(info, dom) {
  const avail = (info.models || []).filter(m => (m.available || m.downloadable) && (!dom || (m.domains || []).includes(dom)));
  if (!avail.length) return null;
  const pref = dom && info.settings && info.settings.default_model[dom];
  const mine = pref && avail.find(m => m.id === pref);
  if (mine) return mine;
  const rankOf = m => (m.rank && dom && m.rank[dom] !== undefined) ? m.rank[dom] : 99;
  return avail.slice().sort((a, b) => rankOf(a) - rankOf(b))[0];
}

export function modelStatus(m) {
  if (m.available) return '';
  if (m.downloadable) return ' (downloads on first use)';
  return m.release === 'preview' ? ' (preview, not published yet)' : ' (not installed)';
}

// One line under the options: which model this job will use and what it is good at.
function modelNote(info) {
  const dom = domain();
  const id = state.runtime.model || (state.plan && state.plan.decision.model);
  const m = info.models.find(x => x.id === id) || autoModel(info, dom);
  if (!m) return null;
  const how = state.runtime.model ? 'Chosen' : 'Automatic';
  return el('p', { class: 'help model-note' }, el('b', {}, `${how}: ${m.title}`), m.role ? ` — ${m.role}.` : '.',
    m.metrics ? el('span', { class: 'muted' }, ' ' + m.metrics) : null);
}

function choices(p) {
  const info = state.info, auto = ['auto', 'Automatic'];
  if (p.name === 'model') return [auto, ...info.models.map(m => [m.id, m.title + (m.role ? ' · ' + m.role : '') + modelStatus(m), !(m.available || m.downloadable)])];
  if (p.name === 'device') return [auto, ...info.machine.devices.map(d => [d, deviceLabel(d)])];
  if (p.name === 'lot') {
    const dom = domain();
    const id = state.runtime.model || (state.plan && state.plan.decision.model);
    const m = info.models.find(x => x.id === id) || autoModel(info, dom);
    const allLots = (m && m.lots) || [];
    const filtered = dom === 'molecule'
      ? allLots.filter(l => MOL_LOTS.has(l))
      : (dom === 'slab' ? allLots.filter(l => !MOL_LOTS.has(l)) : allLots);
    const shown = filtered.length ? filtered : allLots;
    return [auto, ...shown.map(l => [l, lotLabel(l)])];
  }
  const names = { auto: 'Automatic', on: 'On', off: 'Off', float32: 'Single (float32)', float64: 'Double (float64)', ...PRECISION };
  return (p.options || []).map(o => [o, (p.labels && p.labels[o]) || names[o] || o]);
}

function renderOpts() {
  const openMore = optsHost.querySelector('details.more')?.open;
  clear(optsHost);
  let pruned = false, prunedLot = false;
  const buildField = p => {
    const id = 'rt-' + p.name, cur = state.runtime[p.name];
    let ctl;
    if (p.type === 'choice') {
      const opts = choices(p);
      if (cur !== undefined && !opts.some(o => o[0] === cur && !o[2])) {
        delete state.runtime[p.name];
        pruned = true;
        if (p.name === 'lot') prunedLot = true;
      }
      ctl = el('select', { id, dataset: { field: p.name }, title: p.help || null, onchange: () => {
        if (ctl.value === 'auto') delete state.runtime[p.name]; else state.runtime[p.name] = ctl.value;
        if (p.name === 'model') { delete state.runtime.lot; renderOpts(); }
        emit('changed');
      } }, ...opts.map(([v, t, dis]) => el('option', { value: v, selected: v === (state.runtime[p.name] ?? 'auto'), disabled: dis || null }, t)));
    } else {
      const isInt = p.type === 'int';
      ctl = el('input', { id, type: 'number', class: 'num', min: p.min, max: p.max, step: isInt ? 1 : 'any',
        inputmode: isInt ? 'numeric' : 'decimal', value: cur ?? '', placeholder: 'auto',
        dataset: { field: p.name }, title: p.help || null,
        oninput: () => { const t = ctl.value.trim(); if (t === '') delete state.runtime[p.name]; else state.runtime[p.name] = Number(t); emit('changed'); } });
    }
    return el('div', { class: 'field' },
      el('div', { class: 'field-head' }, el('label', { for: id }, p.label),
        p.help ? el('span', { class: 'hint-dot', title: p.help, 'aria-label': p.help }, '?') : null),
      p.unit ? el('div', { class: 'unit-wrap' }, ctl, el('span', { class: 'unit' }, p.unit)) : ctl);
  };
  const all = state.info.catalog.runtime;
  const basic = all.filter(p => !p.advanced), adv = all.filter(p => p.advanced);
  const grid = el('div', { class: 'runtime-grid' }, ...basic.map(buildField));
  optsHost.append(grid);
  const note = modelNote(state.info);
  if (note) optsHost.append(note);
  if (adv.length) {
    const advCustom = adv.filter(p => state.runtime[p.name] !== undefined).length;
    optsHost.append(el('details', { class: 'more', style: 'margin-top:.55rem', open: (openMore || advCustom > 0) || null },
      el('summary', {}, 'Advanced runtime', el('span', { class: 'count' }, advCustom ? `${advCustom} set` : `${adv.length}`)),
      el('div', { class: 'runtime-grid', style: 'margin-top:.4rem' }, ...adv.map(buildField))));
  }
  const n = Object.keys(state.runtime).length;
  optsHost.append(el('div', { style: 'margin-top:.45rem;display:flex;justify-content:flex-end' },
    el('button', { class: 'btn ghost small', type: 'button', disabled: !n || null, onclick: () => { state.runtime = {}; renderOpts(); emit('changed'); } }, 'Reset to automatic')));
  if (prunedLot) toast('Reset level of theory to Automatic for this structure domain.');
  if (pruned && !state.planPending) setTimeout(() => emit('changed'), 0);
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
  on('plan', () => { renderRepro(); renderOpts(); });
  on('structure', ({ slot }) => { if (slot === 1) renderOpts(); });
  on('restore-runtime', renderOpts);
}
