// 1 · Structure card: file drop/chooser, built-in example picker, native SMILES builder, NEB Start/End slot tabs, and charge/spin.
import { el, clear, api, store } from './util.js';
import { state, emit, on, task } from './state.js';
import { renderStructBar, renderSystem } from './structcard.js';
import { formulaText } from './labels.js';

const EXT = { xyz: 'xyz', extxyz: 'extxyz', cif: 'cif', vasp: 'vasp', poscar: 'vasp', contcar: 'vasp' };

export function formatOf(name) {
  const base = name.split(/[\\/]/).pop(), low = base.toLowerCase(), ext = low.includes('.') ? low.split('.').pop() : '';
  if (EXT[ext]) return EXT[ext];
  if (/^(poscar|contcar)/.test(low)) return 'vasp';
  return null;
}

let errBox1, errBox2, pick1, pick2, smilesInp1, smilesInp2, slot1Wrap, slot2Wrap, slotTabs, systemBox, barHost, activeSlot = 1;

export function setCompact() { /* Layout is now viewport-fitted 2-column; no manual fold required. */ }

// slot 1 = primary structure, slot 2 = end structure of a reaction barrier (NEB).
export async function loadStructure(s, slot = 1, opts = {}) {
  const err = slot === 1 ? errBox1 : errBox2;
  if (err) err.hidden = true;
  try {
    const r = await api('POST', '/api/structure', { structure: { format: s.format, text: s.text, name: s.name } });
    if (slot === 1) {
      state.structure = s; state.analysis = r.analysis; state.atoms = r.viewer;
      if (!opts.keepSystem) {
        if (r.analysis && r.analysis.domain === 'molecule') {
          const q = r.analysis.charge_hint ?? 0;
          state.system = { charge: q, spin: r.analysis.spin_hint ?? ((r.analysis.electrons - q) % 2 ? 2 : 1) };
        } else {
          state.system = {};
        }
      }
    } else {
      state.structure2 = s; state.analysis2 = r.analysis; state.atoms2 = r.viewer;
    }
    if (s.format === 'builtin' && slot === 1) store('example', s.text);
    syncPickers();
    emit('structure', { slot, keepViewer: !!opts.keepViewer });
    if (slot === 2 && !opts.keepViewer) emit('show-atoms', 2);
    return true;
  } catch (e) {
    if (err) { err.textContent = e.message; err.hidden = false; }
    return false;
  }
}

function examplePicker(slot) {
  const groups = new Map();
  state.info.examples.forEach(x => { let g = groups.get(x.group); if (!g) groups.set(x.group, g = []); g.push(x); });
  const sel = el('select', { 'aria-label': slot === 1 ? 'Example structure' : 'Example end structure',
    onchange: () => { const x = state.info.examples.find(q => q.id === sel.value); if (x) loadStructure({ format: 'builtin', text: x.id, name: x.id, title: x.title }, slot); } },
    el('option', { value: '' }, 'Choose an example…'),
    ...[...groups].map(([g, xs]) => el('optgroup', { label: g }, ...xs.map(x => el('option', { value: x.id }, x.title)))));
  return sel;
}

function dropZone(slot) {
  const input = el('input', { type: 'file', accept: '.xyz,.extxyz,.cif,.vasp,.poscar,.contcar,POSCAR,CONTCAR', hidden: true,
    onchange: () => { if (input.files[0]) readFile(input.files[0], slot); input.value = ''; } });
  const zone = el('div', { class: 'drop', tabindex: 0, role: 'button',
    'aria-label': slot === 1 ? 'Choose a structure file' : 'Choose the end structure file',
    onclick: () => input.click(), onkeydown: e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } } },
    el('div', { class: 'drop-main' }, slot === 1 ? 'Drop CIF, XYZ or POSCAR file' : 'Drop end-structure file'),
    el('div', { class: 'drop-sub' }, 'or click to browse'), input);
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('over'); const f = e.dataTransfer.files[0]; if (f) readFile(f, slot); });
  return zone;
}

function readFile(file, slot) {
  const fmt = formatOf(file.name), err = slot === 1 ? errBox1 : errBox2;
  if (!fmt) { err.textContent = `“${file.name}” is not supported. Use .cif, .xyz, .extxyz, .vasp, POSCAR or CONTCAR.`; err.hidden = false; return; }
  if (file.size > 8e6) { err.textContent = 'The file is larger than 8 MB.'; err.hidden = false; return; }
  const rd = new FileReader();
  rd.onload = () => loadStructure({ format: fmt, text: String(rd.result), name: file.name.slice(0, 80), title: file.name }, slot);
  rd.onerror = () => { err.textContent = 'The file could not be read.'; err.hidden = false; };
  rd.readAsText(file);
}

function smilesField(slot = 1) {
  const inp = el('input', {
    type: 'text',
    placeholder: slot === 1 ? 'e.g. CC(=O)Oc1ccccc1C(=O)O' : 'End-state SMILES (same atom order)',
    'aria-label': slot === 1 ? 'SMILES string' : 'End structure SMILES string',
    spellcheck: 'false', autocomplete: 'off', class: 'mono',
  });
  if (slot === 1) smilesInp1 = inp; else smilesInp2 = inp;
  const go = () => {
    const t = inp.value.trim().replace(/^smiles:\s*/i, '');
    if (t) loadStructure({ format: 'smiles', text: t, name: 'smiles', title: t }, slot);
  };
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
  return el('div', { class: 'field' },
    el('div', { class: 'field-head' }, el('label', { class: 'lbl' }, 'Or build from SMILES'),
      el('span', { class: 'hint-dot', title: 'Generates a 3D molecular conformer directly from an OpenSMILES string (molecules, ions & radicals).' }, '?')),
    el('div', { class: 'smiles-row' }, inp, el('button', { class: 'btn', type: 'button', onclick: go }, 'Build')));
}

function syncPickers() {
  if (pick1) pick1.value = state.structure && state.structure.format === 'builtin' ? state.structure.text : '';
  if (pick2) pick2.value = state.structure2 && state.structure2.format === 'builtin' ? state.structure2.text : '';
  if (smilesInp1) smilesInp1.value = state.structure && state.structure.format === 'smiles' ? state.structure.text : '';
  if (smilesInp2) smilesInp2.value = state.structure2 && state.structure2.format === 'smiles' ? state.structure2.text : '';
}

function setSlot(slot, showViewer = true) {
  const needs2 = !!(task() && task().needs_second);
  activeSlot = needs2 ? slot : 1;
  slotTabs.hidden = !needs2;
  slot1Wrap.hidden = activeSlot !== 1;
  slot2Wrap.hidden = activeSlot !== 2;
  clear(slotTabs);
  if (needs2) {
    const f1 = state.analysis ? ` (${formulaText(state.analysis.formula)})` : '';
    const f2 = state.analysis2 ? ` (${formulaText(state.analysis2.formula)})` : ' · needed';
    slotTabs.append(
      el('button', { type: 'button', 'aria-selected': String(activeSlot === 1), onclick: () => setSlot(1) }, `Start${f1}`),
      el('button', { type: 'button', class: !state.analysis2 && activeSlot !== 2 ? 'needs' : '', 'aria-selected': String(activeSlot === 2), onclick: () => setSlot(2) }, `End${f2}`));
  }
  if (showViewer && ((activeSlot === 1 && state.atoms) || (activeSlot === 2 && state.atoms2))) emit('show-atoms', activeSlot);
}

export function mountStructure(host, structBarHost) {
  barHost = structBarHost || document.getElementById('struct-bar');
  clear(host);
  errBox1 = el('div', { class: 'msg error', role: 'alert', hidden: true });
  errBox2 = el('div', { class: 'msg error', role: 'alert', hidden: true });
  pick1 = examplePicker(1); pick2 = examplePicker(2);
  systemBox = el('div');
  slotTabs = el('div', { class: 'seg seg-full slot-tabs', role: 'tablist', 'aria-label': 'Reaction endpoints', hidden: true });

  slot1Wrap = el('div', {},
    dropZone(1),
    el('div', { class: 'struct-inputs' },
      el('div', { class: 'field' }, el('div', { class: 'field-head' }, el('label', { class: 'lbl' }, 'Or choose an example')), pick1),
      smilesField(1)),
    errBox1);

  slot2Wrap = el('div', { hidden: true },
    el('p', { class: 'task-desc', style: 'margin:0 0 .45rem' }, 'Same atoms in the same order as the start structure.'),
    dropZone(2),
    el('div', { class: 'struct-inputs' },
      el('div', { class: 'field' }, el('div', { class: 'field-head' }, el('label', { class: 'lbl' }, 'Or choose an example')), pick2),
      smilesField(2)),
    errBox2);

  host.append(el('h2', { class: 'panel-title' }, '1 · Structure'), slotTabs, slot1Wrap, slot2Wrap, systemBox);

  on('structure', ({ slot }) => {
    setSlot(activeSlot, false);
    if (slot === 1) renderSystem(systemBox);
    if (barHost) renderStructBar(barHost);
  });
  on('task', () => {
    syncPickers();
    setSlot(task() && task().needs_second && !state.analysis2 ? 2 : 1, false);
    if (barHost) renderStructBar(barHost);
  });
  setSlot(1, false);
  renderSystem(systemBox);
  if (barHost) renderStructBar(barHost);
}
