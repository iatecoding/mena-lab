// 2 · Calculation: compact 2x3 task grid (no truncated subtitles) + concise 2-column parameter grid.
import { el, clear, store, toast } from './util.js';
import { state, emit, on, task, domain, value, setValue, visible, taskAllowed } from './state.js';
import { DOMAIN_PLURAL } from './labels.js';

export const SHORT_TITLES = {
  singlepoint: 'Energy & forces',
  relax: 'Relax structure',
  md: 'Dynamics (MD)',
  vibrations: 'Vibrations & IR',
  phonons: 'Phonons',
  neb: 'Reaction barrier',
};

let stripHost, descHost, formHost;

export function shownTasks() { return state.info.catalog.tasks.filter(t => !t.hidden); }

export function selectTask(id) {
  if (!shownTasks().some(t => t.id === id)) return;
  state.task = id; store('task', id);
  renderStrip(); renderForm();
  emit('task'); emit('changed');
}

function renderStrip() {
  clear(stripHost);
  const dom = domain();
  shownTasks().forEach((t, i) => {
    const ok = taskAllowed(t, dom), sel = t.id === state.task;
    const why = ok ? '' : `Only for ${t.domains.map(d => DOMAIN_PLURAL[d] || d).join(' and ')}.`;
    const b = el('button', {
      class: 'tstep' + (sel ? ' selected' : '') + (ok ? '' : ' off'), type: 'button', role: 'radio', 'aria-checked': String(sel),
      'aria-disabled': ok ? null : 'true', title: ok ? t.summary : `${t.title} — ${why}`,
      onclick: () => { if (ok) selectTask(t.id); },
    }, el('span', { class: 'tstep-n' }, String(i + 1).padStart(2, '0')),
    el('span', { class: 'tstep-title' }, SHORT_TITLES[t.id] || t.title));
    stripHost.append(b);
  });
  const cur = task();
  if (descHost) descHost.textContent = cur ? cur.summary : '';
}

function keys(e) {
  const k = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
  if (!k) return;
  const btns = [...stripHost.querySelectorAll('.tstep')], i = btns.indexOf(document.activeElement);
  if (i < 0) return;
  e.preventDefault();
  for (let j = 1; j <= btns.length; j++) {
    const b = btns[(i + k * j + btns.length * 2) % btns.length];
    if (!b.classList.contains('off')) { b.focus(); b.click(); break; }
  }
}

export function paramInput(p, v, onSet, opts = {}) {
  const id = (opts.prefix || 'p-') + p.name;
  const common = { id, title: p.help || null, dataset: { field: p.name } };
  const head = el('div', { class: 'field-head' },
    p.type === 'bool' ? el('span', { class: 'lbl', id: id + '-l' }, p.label) : el('label', { for: id }, p.label),
    p.help ? el('span', { class: 'hint-dot', title: p.help, 'aria-label': p.help }, '?') : null);
  const wide = opts.wide ? ' span-2' : '';
  if (p.type === 'bool') {
    const seg = el('div', { ...common, class: 'seg seg-full', role: 'group', 'aria-labelledby': id + '-l' },
      ...[[true, 'On'], [false, 'Off']].map(([val, txt]) =>
        el('button', { type: 'button', 'aria-selected': String(!!v === val), onclick: () => { if (!!v !== val) onSet(val, true); } }, txt)));
    return el('div', { class: 'field' + wide }, head, seg);
  }
  let ctl;
  if (p.options) {
    ctl = el('select', { ...common, onchange: () => onSet(p.type === 'choice' ? ctl.value : Number(ctl.value), true) },
      ...p.options.map(o => el('option', { value: o, selected: o === v }, (p.labels && p.labels[o]) || (typeof o === 'number' ? o.toLocaleString('en-US') : o))));
  } else {
    ctl = el('input', { ...common, type: 'number', class: 'num', value: v ?? '', min: p.min, max: p.max, step: p.type === 'int' ? 1 : 'any',
      inputmode: p.type === 'int' ? 'numeric' : 'decimal', oninput: () => { const t = ctl.value.trim(); onSet(t === '' ? null : Number(t), false); } });
  }
  return el('div', { class: 'field' + wide }, head, p.unit ? el('div', { class: 'unit-wrap' }, ctl, el('span', { class: 'unit' }, p.unit)) : ctl);
}

function buildGrid(params, mk) {
  const g = el('div', { class: 'param-grid' });
  params.forEach((p, idx) => {
    const wide = params.length === 1 || (params.length % 2 === 1 && idx === 0 && p.options);
    g.append(mk(p, wide));
  });
  return g;
}

function renderForm(focusName) {
  const t = task(), dom = domain();
  const openMore = formHost.querySelector('details.more')?.open;
  clear(formHost);
  if (!t) return;
  const shown = t.params.filter(p => visible(p, t.params, q => value(q), dom));
  if (!shown.length) return;
  const basic = shown.filter(p => !p.advanced), adv = shown.filter(p => p.advanced);
  const mk = (p, wide) => paramInput(p, value(p), (v, redraw) => { setValue(p, v); emit('changed'); if (redraw) renderForm(p.name); }, { wide });
  if (basic.length) formHost.append(buildGrid(basic, mk));
  if (adv.length) {
    formHost.append(el('details', { class: 'more task-adv', open: openMore || null },
      el('summary', {}, 'Advanced settings', el('span', { class: 'count' }, `${adv.length}`)),
      buildGrid(adv, mk)));
  }
  if (focusName) { const f = formHost.querySelector(`[data-field="${focusName}"]`); if (f) (f.querySelector('[aria-selected="true"]') || f).focus(); }
}

export function markInvalid(field) {
  document.querySelectorAll('[aria-invalid]').forEach(n => n.removeAttribute('aria-invalid'));
  if (!field) return;
  const n = document.querySelector(`#view-workspace [data-field="${CSS.escape(field)}"]`);
  if (n) n.setAttribute('aria-invalid', 'true');
}

export function mountTasks(sh, fh, dh) {
  stripHost = sh; formHost = fh; descHost = dh || document.getElementById('task-desc');
  if (!task() || task().hidden) state.task = 'singlepoint';
  stripHost.addEventListener('keydown', keys);
  renderStrip(); renderForm();
  on('structure', ({ slot }) => {
    if (slot !== 1) return;
    const dom = domain(), cur = task();
    if (cur && dom && !taskAllowed(cur, dom)) {
      const swap = cur.id === 'vibrations' ? 'phonons' : cur.id === 'phonons' ? 'vibrations' : 'singlepoint';
      const next = shownTasks().find(t => t.id === swap && taskAllowed(t, dom)) || shownTasks().find(t => taskAllowed(t, dom));
      if (next && next.id !== state.task) {
        state.task = next.id; store('task', next.id);
        toast(`Switched task to ${SHORT_TITLES[next.id] || next.title} for this ${dom}.`);
        emit('task');
      }
    }
    renderStrip(); renderForm();
  });
  on('restore-task', () => { renderStrip(); renderForm(); emit('task'); });
}
