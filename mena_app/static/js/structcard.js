// Compact structure bar (top of right stage) and molecule charge/spin controls (inside Structure card).
import { el, clear, fmt } from './util.js';
import { state, emit, task } from './state.js';
import { DOMAIN, formulaNode } from './labels.js';

function sourceText(s) {
  if (!s) return '';
  if (s.format === 'builtin') return (s.title || (state.info.examples.find(x => x.id === s.text) || {}).title || s.text);
  if (s.format === 'smiles') return `SMILES: ${s.text}`;
  return s.title || s.name || '';
}

function dot() { return el('span', { class: 'sbar-dot', 'aria-hidden': 'true' }, '·'); }

export function renderStructBar(host) {
  clear(host);
  const a = state.analysis, needs2 = !!(task() && task().needs_second);
  if (!a) {
    host.append(el('span', { class: 'muted' }, 'No structure loaded — choose an example, drop a file, or enter a SMILES string.'));
    return;
  }
  const meta = el('div', { class: 'sbar-meta' });
  const src = sourceText(state.structure);
  if (src) meta.append(el('b', {}, src), dot());
  meta.append(el('span', {}, DOMAIN[a.domain] || a.domain), dot(), el('span', { class: 'num' }, `${fmt(a.n_atoms, 0)} atom${a.n_atoms === 1 ? '' : 's'}`));
  if (a.spacegroup) meta.append(dot(), el('span', {}, a.spacegroup));
  if (a.cell) {
    const c = a.cell;
    meta.append(dot(), el('span', { class: 'num', title: `Angles: ${c.slice(3).map(v => fmt(v, 1)).join('°, ')}°` },
      `a,b,c = ${c.slice(0, 3).map(v => fmt(v, 2)).join(', ')} Å`));
  }
  if (a.density) meta.append(dot(), el('span', { class: 'num' }, `${fmt(a.density, 2)} g/cm³`));
  if (a.domain === 'molecule') meta.append(dot(), el('span', { class: 'num' }, `${fmt(a.electrons, 0)} e⁻`));

  host.append(el('span', { class: 'sbar-formula' }, formulaNode(a.formula)), meta);

  if (needs2) {
    const a2 = state.analysis2;
    const right = el('div', { class: 'sbar-right' },
      el('span', { class: 'muted' }, '→ End: '),
      a2 ? el('b', {}, formulaNode(a2.formula)) : el('span', { class: 'pill warn' }, 'Select end structure'),
      el('button', { class: 'btn ghost small', type: 'button', onclick: () => emit('show-atoms', 1) }, 'View start'),
      a2 ? el('button', { class: 'btn ghost small', type: 'button', onclick: () => emit('show-atoms', 2) }, 'View end') : null);
    host.append(right);
  }

  const warns = [...(a.warnings || []), ...((needs2 && state.analysis2 && state.analysis2.warnings) || [])];
  if (warns.length) {
    host.append(el('div', { class: 'sbar-warns' }, warns.map(w => el('div', { class: 'warnbox', role: 'note' }, w))));
  }
}

// Charge and spin multiplicity for molecules (concise labels + hover tooltips instead of multi-line paragraphs).
export function renderSystem(host) {
  clear(host);
  const a = state.analysis;
  if (!a || a.domain !== 'molecule') return;
  if (state.system.charge === undefined) state.system.charge = a.charge_hint ?? 0;
  if (state.system.spin === undefined) state.system.spin = a.spin_hint ?? ((a.electrons - (state.system.charge ?? 0)) % 2 ? 2 : 1);
  const shortLabel = { charge: 'Charge (e)', spin: 'Spin (2S+1)' };
  const fields = state.info.catalog.system.filter(p => !p.domains || p.domains.includes(a.domain)).map(p => {
    const id = 'sys-' + p.name;
    const inp = el('input', { id, type: 'number', step: 1, min: p.min, max: p.max, value: state.system[p.name] ?? p.default, class: 'num',
      title: p.help || null, dataset: { field: p.name },
      oninput: () => { const v = inp.value.trim(); state.system[p.name] = v === '' ? null : Number(v); emit('changed'); } });
    return el('div', { class: 'field' },
      el('div', { class: 'field-head' }, el('label', { for: id }, shortLabel[p.name] || p.label),
        p.help ? el('span', { class: 'hint-dot', title: p.help, 'aria-label': p.help }, '?') : null),
      inp);
  });
  host.append(el('div', { class: 'system-grid' }, fields));
}
