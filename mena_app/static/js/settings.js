// Settings dialog: the default model for molecules, crystals and surfaces (used when the model is "Automatic").
// Saved per user by the server in ~/.mena/settings.json, so the CLI uses the same defaults (`mena models default`).
import { el, clear, api, toast } from './util.js';
import { state, emit } from './state.js';

const KINDS = [['molecule', 'Molecules'], ['crystal', 'Crystals'], ['slab', 'Surfaces']];
let dlg, body;

function recommended(dom) {
  const ok = (state.info.models || []).filter(m => (m.available || m.downloadable) && (m.domains || []).includes(dom));
  const rank = m => (m.rank && m.rank[dom] !== undefined ? m.rank[dom] : 99);
  return ok.sort((a, b) => rank(a) - rank(b))[0] || null;
}

function status(m) {
  if (m.available) return '';
  if (m.downloadable) return ` (downloads on first use, ${m.size_mb} MB)`;
  return m.release === 'preview' ? ' (preview, not published yet)' : ' (not installed)';
}

function render() {
  clear(body);
  const cur = (state.info.settings && state.info.settings.default_model) || {};
  KINDS.forEach(([dom, label]) => {
    const rec = recommended(dom);
    const id = 'set-' + dom;
    const choices = (state.info.models || []).filter(m => (m.domains || []).includes(dom));
    const sel = el('select', { id, onchange: () => save(dom, sel.value) },
      el('option', { value: 'auto', selected: !cur[dom] || null }, `Automatic${rec ? ': ' + rec.title : ''}`),
      ...choices.map(m => el('option', { value: m.id, selected: cur[dom] === m.id || null, disabled: !(m.available || m.downloadable) || null },
        m.title + (m.role ? ' · ' + m.role : '') + status(m))));
    body.append(el('div', { class: 'field' }, el('label', { for: id }, label), sel));
  });
  body.append(el('p', { class: 'help' }, 'Used when the model is set to Automatic. Each job can still pick another model under Tune. ',
    'The command line uses the same choice: ', el('code', {}, 'mena models default crystal lam6-v1'), '.'));
}

async function save(dom, value) {
  try {
    const r = await api('POST', '/api/settings', { default_model: { [dom]: value } });
    state.info.settings = r.settings;
    state.info.models = r.models;
    render();
    emit('changed');
  } catch (e) {
    toast(e.message, 'error');
    render();
  }
}

export function mountSettings() {
  const btn = document.getElementById('settings-btn');
  dlg = el('dialog', { class: 'settings-dlg', 'aria-labelledby': 'settings-title' });
  body = el('div', {});
  dlg.append(
    el('div', { class: 'drawer-head' }, el('h2', { id: 'settings-title', class: 'panel-title', style: 'margin:0' }, 'Default models'),
      el('button', { type: 'button', class: 'btn ghost small', onclick: () => dlg.close() }, 'Done')),
    body);
  document.body.append(dlg);
  btn.addEventListener('click', () => { render(); dlg.showModal(); });
}
