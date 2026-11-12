// Pinned bottom dock: compact 2-line execution plan, expandable Tune / Code / Why drawer, and always-visible Run button.
import { el, clear, api, fmt, debounce, approx } from './util.js';
import { state, emit, on, buildJob, task, domain } from './state.js';
import { lotLabel, deviceLabel, DTYPE } from './labels.js';
import { markInvalid, SHORT_TITLES } from './tasks.js';

let host, whyHost, drawer, drawerTitle, runBtn, runLabel, cancelBtn, errBox, activeDrawer = null, seq = 0;

const SHORT_DTYPE = { float32: 'fp32', float64: 'fp64' };
const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || '');

export function toggleDrawer(mode) {
  activeDrawer = activeDrawer === mode ? null : mode;
  syncDrawer();
  render();
}

function syncDrawer() {
  if (!drawer) return;
  drawer.hidden = !activeDrawer;
  const titles = { tune: 'Customize execution', code: 'Reproduce (CLI / Python / JSON)', why: 'Why auto chose these settings' };
  if (drawerTitle && activeDrawer) drawerTitle.textContent = titles[activeDrawer] || '';
  const optsEl = document.getElementById('runtime-opts');
  const reproEl = document.getElementById('repro');
  if (whyHost) whyHost.hidden = activeDrawer !== 'why';
  if (optsEl) optsEl.hidden = activeDrawer !== 'tune';
  if (reproEl) reproEl.hidden = activeDrawer !== 'code';
}

export async function refreshPlan() {
  const job = buildJob(), my = ++seq;
  if (!job) { state.plan = null; state.planError = state.structure ? null : 'Choose a structure first.'; render(); return; }
  if (task().needs_second && !state.structure2) { state.plan = null; state.planError = 'Select the end structure in 1 · Structure above.'; render(); return; }
  state.planPending = true; render();
  try {
    const r = await api('POST', '/api/plan', job);
    if (my !== seq) return;
    state.plan = r; state.planError = null; markInvalid(null);
  } catch (e) {
    if (my !== seq) return;
    state.plan = null; state.planError = e.message; markInvalid(e.field);
  }
  state.planPending = false;
  render(); emit('plan');
}
const later = debounce(refreshPlan, 250);

function lotShort(d) {
  if (d.lot && d.lot !== 'auto') return lotLabel(d.lot);
  return domain() === 'molecule' ? 'ωB97M-V' : 'PBE / PBE+U';
}

function row(label, val, reason) {
  return el('div', { class: 'prow' }, el('div', { class: 'pk' }, label), el('div', { class: 'pv' }, val),
    reason ? el('div', { class: 'pr' }, reason) : null);
}

function renderWhy(p) {
  if (!whyHost) return;
  clear(whyHost);
  if (!p) { whyHost.append(el('p', { class: 'muted', style: 'margin:0' }, 'Select a valid structure and settings first.')); return; }
  const d = p.decision, m = p.model || {};
  whyHost.append(
    row('Model', m.title || d.model, (d.reasons && d.reasons.model) || m.metrics),
    row('Theory', lotShort(d), d.reasons && d.reasons.lot),
    row('Device', deviceLabel(d.device), d.reasons && d.reasons.device),
    row('Precision', DTYPE[d.dtype] || d.dtype, d.reasons && ((d.reasons.dtype || '') + (d.low_memory ? ' Low-memory mode is on.' : ''))),
    row('Compiled', d.compile ? 'On' : 'Off', d.reasons && d.reasons.compile),
    row('Est. time', d.seconds == null ? 'Not calibrated' : approx(d.seconds),
      d.seconds == null ? 'Run mena calibrate once for hardware timing.' : `${fmt(d.calls, 0)} call${d.calls === 1 ? '' : 's'} on ${fmt(d.atoms, 0)} atoms.`),
    row('Memory', d.peak_gb != null && d.free_gb ? `${fmt(d.peak_gb, 2)} of ${fmt(d.free_gb, 1)} GB free` : 'Uncalibrated',
      d.reasons && d.reasons.low_memory));
}

function sep() { return el('span', { class: 'sep', 'aria-hidden': 'true' }, '·'); }

function render() {
  clear(host);
  const p = state.plan, nCustom = Object.keys(state.runtime || {}).length;
  host.classList.toggle('pending', !!state.planPending);
  const head = el('div', { class: 'plan-head' },
    el('span', { class: 'plan-tag' + (nCustom ? ' custom' : '') }, nCustom ? `Custom plan (${nCustom})` : 'Auto plan'),
    el('div', { class: 'plan-links' },
      el('button', { type: 'button', class: 'dock-link' + (activeDrawer === 'tune' ? ' active' : ''), onclick: () => toggleDrawer('tune') },
        nCustom ? `Tune (${nCustom})` : 'Tune'),
      el('button', { type: 'button', class: 'dock-link' + (activeDrawer === 'code' ? ' active' : ''), onclick: () => toggleDrawer('code') }, 'CLI / Python'),
      el('button', { type: 'button', class: 'dock-link' + (activeDrawer === 'why' ? ' active' : ''), onclick: () => toggleDrawer('why') }, 'Why?')));
  host.append(head);

  if (!p) {
    host.append(el('div', { class: 'plan-line muted' }, state.planPending ? 'Planning…' : (state.planError ? 'Adjust settings below to run.' : 'Waiting for structure…')));
  } else {
    const d = p.decision, m = p.model || {};
    const line1 = el('div', { class: 'plan-line' },
      el('span', { class: 'plan-strong', title: [m.training, m.metrics].filter(Boolean).join('\n') }, m.title || d.model),
      sep(), el('span', { title: d.reasons && d.reasons.lot }, lotShort(d)),
      sep(), el('span', { title: d.reasons && d.reasons.device }, `${deviceLabel(d.device)} ${SHORT_DTYPE[d.dtype] || d.dtype}`));
    if (d.low_memory) line1.append(sep(), el('span', { class: 'muted' }, 'low-mem'));
    if (d.compile) line1.append(sep(), el('span', { class: 'muted' }, 'compiled'));

    const timeNode = d.seconds == null
      ? el('span', { class: 'muted', title: 'Run mena calibrate once for exact hardware estimates' }, `${fmt(d.calls, 0)} call${d.calls === 1 ? '' : 's'}`)
      : el('span', { class: 'num plan-strong', title: `${fmt(d.calls, 0)} model call${d.calls === 1 ? '' : 's'} on ${fmt(d.atoms, 0)} atoms` }, approx(d.seconds));
    const line2 = el('div', { class: 'plan-line muted' }, el('span', {}, 'Est.'), timeNode);
    if (d.peak_gb != null && d.free_gb) {
      const pct = Math.min(100, 100 * d.peak_gb / d.free_gb);
      line2.append(el('span', { class: 'plan-mem num', title: `Estimated peak memory: ${fmt(d.peak_gb, 2)} GB of ${fmt(d.free_gb, 1)} GB free` },
        `${fmt(d.peak_gb, 2)} / ${fmt(d.free_gb, 0)} GB`,
        el('span', { class: 'membar ' + (d.memory_level || '') }, el('i', { style: `width:${Math.max(3, pct)}%` }))));
    }
    host.append(line1, line2);
  }
  renderWhy(p);

  const curTask = task();
  if (runLabel) runLabel.textContent = curTask ? `Run ${SHORT_TITLES[curTask.id] || curTask.title}` : 'Run';
  const busy = !!state.running;
  errBox.hidden = !state.planError || busy;
  errBox.textContent = state.planError || '';
  runBtn.disabled = busy || !p || !!state.planError || (p && p.decision.memory_level === 'refuse');
  runBtn.hidden = busy;
  cancelBtn.hidden = !busy;
}

export function setRunning(on_) { render(); if (!on_) { cancelBtn.disabled = false; cancelBtn.textContent = '■ Stop'; } }

export function mountPlan(planHost, actionsHost) {
  host = planHost;
  whyHost = document.getElementById('plan-why');
  drawer = document.getElementById('dock-drawer');
  drawerTitle = document.getElementById('drawer-title');
  const closeBtn = document.getElementById('drawer-close');
  if (closeBtn) closeBtn.addEventListener('click', () => { activeDrawer = null; syncDrawer(); render(); });

  errBox = el('div', { class: 'msg error', role: 'alert', 'aria-live': 'polite', hidden: true });
  runLabel = el('span', {}, 'Run');
  runBtn = el('button', { class: 'btn primary run', type: 'button', disabled: true, title: 'Run calculation (Cmd/Ctrl + Enter)', onclick: () => emit('run') },
    el('span', { 'aria-hidden': 'true' }),
    el('span', { class: 'run-main' }, '▶ ', runLabel),
    el('span', { class: 'kbd', 'aria-hidden': 'true' }, IS_MAC ? '⌘↵' : 'Ctrl↵'));
  cancelBtn = el('button', { class: 'btn danger run', type: 'button', hidden: true,
    onclick: () => { cancelBtn.disabled = true; cancelBtn.textContent = 'Stopping…'; emit('cancel'); } },
    el('span', { class: 'run-main' }, '■ Stop'));
  actionsHost.append(errBox, runBtn, cancelBtn);

  window.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !document.getElementById('view-workspace').hidden && !runBtn.disabled && !state.running) {
      e.preventDefault();
      emit('run');
    }
  });

  on('changed', later);
  on('structure', later);
  on('task', () => render());
  on('running', () => render());
  syncDrawer();
  render();
}
